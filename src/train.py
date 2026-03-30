import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import sys
import datetime
import time
import math
import numpy as np
from skimage import segmentation
from tqdm import tqdm

# 引入自定义的数据集类和工具函数
# 添加新的 CAVE 数据集类
from hsi_dataset import TrainDataset, ValidDataset, CAVETrainDataset, CAVEValidDataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM, Loss_SSIM, time2file_name

sys.path.append("..")
# 引入模型生成器
from model import model_generator

# =============================================================================
# 1. 参数解析部分
# =============================================================================
parser = argparse.ArgumentParser(description="DL4sSR Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='SSDCN') 
parser.add_argument('--pretrained_model_path', type=str, default=None) 
parser.add_argument("--batch_size", type=int, default=4, help="batch size") 
parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps") 
parser.add_argument("--threads", type=int, default=4, help="threads") 
parser.add_argument("--start_epoch", type=int, default=1, help="start of epochs")
parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs") 
parser.add_argument("--save_epoch", type=int, default=10, help="save of epochs") 
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--outf", type=str, default='checkpoint/', help='path log files') 
parser.add_argument("--data_root", type=str, default='dataset/') 
parser.add_argument("--patch_size", type=int, default=128, help="patch size") 
parser.add_argument("--stride", type=int, default=32, help="stride") 
parser.add_argument("--gpus", type=str, default='0', help='gpu name') 
parser.add_argument("--dataset", type=str, default='CAVE', choices=['NTIRE', 'CAVE'], help="dataset name")
# [新增] 损失函数选择
parser.add_argument("--loss_mode", type=str, default='L1', choices=['L1', 'MRAE', 'L1_MRAE'], help="Training loss function")

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

# =============================================================================
# 2. 模型与数据准备
# =============================================================================

# 初始化模型
model = model_generator(opt.method, opt.pretrained_model_path)

# 生成保存路径
filepath = opt.dataset + "_" + opt.method + "_b" + str(opt.batch_size*opt.accumulation_steps) + "_" + opt.loss_mode + "_lr" + str(opt.lr)+'/'
print(f"Checkpoint folder: {filepath}")
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# 加载数据集
print("\nloading dataset ...")
if opt.dataset == 'NTIRE':
    # NTIRE 配置：保持原项目逻辑
    train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
    val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
    
    # [NTIRE 特有] 硬编码 Iteration
    per_epoch_iteration = 5000
    
elif opt.dataset == 'CAVE':
    # CAVE 配置：SOTA 逻辑
    train_data = CAVETrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, arg=True, stride=opt.stride)
    val_data = CAVEValidDataset(data_root=opt.data_root)
    
    # [CAVE 特有] 调整 Iteration (降低一个Epoch的量以加快更新频率)
    # 不再强制需要跑 4000 个 iteration（因为现在用的是 Random Crop，随便多少都行）
    # 我们把一个 Epoch 设置成 1000 次 iteration，这样验证和保存日志会更频繁
    per_epoch_iteration = 1000

print("Validation set samples: ", len(val_data))
total_iteration = per_epoch_iteration * opt.end_epoch
print(f"Real dataset length: {len(train_data)}")
print(f"Iteration per epoch: {per_epoch_iteration}")
print(f"Total iterations: {total_iteration}")

# 定义损失函数
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_ssim = Loss_SSIM()
criterion_l1 = nn.L1Loss(reduction='mean')

# 设置输出目录
file_log = opt.outf
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + filepath + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_sam.cuda()
    criterion_ssim.cuda()
    criterion_l1.cuda()
    print('GPU {} is used!'.format(opt.gpus))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
# 切换一下优化器和学习率调度器的设置
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), eps=1e-8)
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iteration, power=1.5)

log_dir = os.path.join(file_log, 'train.log')
logger = initialize_logger(log_dir)
logger.info("Dataset[%s], Loss[%s], Batch[%d], Patch[%d], Stride[%d]" % (
    opt.dataset, opt.loss_mode, opt.batch_size*opt.accumulation_steps, opt.patch_size, opt.stride))

# =============================================================================
# HPRN 模型特有的 SLIC 语义标签生成函数
# =============================================================================
def generate_slic_semantic_labels(images_tensor, scales=[8, 12, 16, 20]):
    """
    根据给定的 RGB Tensor 动态生成 SLIC 语义标签。
    images_tensor: [B, C, H, W] 的 CUDA Tensor
    返回: [B, len(scales), H, W] 的语义标签 CUDA Tensor
    """
    b, c, h, w = images_tensor.shape
    device = images_tensor.device
    
    # 转换回 numpy 以供 skimage 运算
    images_np = images_tensor.detach().cpu().numpy()
    # 转换格式 [B, C, H, W] -> [B, H, W, C]
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    batch_labels = []
    for i in range(b):
        img_uint8 = np.uint8(np.clip(images_np[i] * 255.0, 0, 255))
        semantic_label_list = []
        for s in scales:
            label = segmentation.slic(img_uint8, start_label=1, n_segments=s)
            semantic_label_list.append(label[None, :])
        # [4, H, W]
        img_labels = np.concatenate(semantic_label_list, axis=0)
        batch_labels.append(img_labels[None, :])
    
    # [B, 4, H, W]
    batch_labels_np = np.concatenate(batch_labels, axis=0)
    return torch.from_numpy(batch_labels_np).float().to(device)


# =============================================================================
# 3. 主训练循环
# =============================================================================
def main():
    cudnn.benchmark = True 
    record_mrae_loss = 1000 
    epoch = 0
    iteration = 0
    
    # 初始化混合精度 Scaler (针对 GPU 防爆显存加提速)
    scaler = torch.amp.GradScaler('cuda')
    
    # 验证集 Loader
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=opt.threads, pin_memory=True)
    
    # 训练集 Loader
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, 
                              num_workers=opt.threads, pin_memory=True, drop_last=True)

    while iteration < total_iteration:
        epoch += 1
        model.train()
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, total=per_epoch_iteration, desc=f"Epoch {epoch}/{opt.end_epoch}")
        
        for i, (images, labels) in enumerate(pbar):
            # 加上 non_blocking=True 加速数据传输到 GPU
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            
            lr = optimizer.param_groups[0]['lr']
            
            # 启用 GPU 混合精度前向传播
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if opt.method == 'HPRN':
                    # HPRN 模型需要额外的 semantic_label
                    # 动态生成 SLIC 标签（非常耗时，如果在 __getitem__ 做会更好，但这里快速兼容）
                    semantic_labels = generate_slic_semantic_labels(images)
                    output = model(images, semantic_labels)
                else:
                    output = model(images)
                
                # [核心修改] 根据参数选择损失函数
                if opt.loss_mode == 'L1':
                    loss = criterion_l1(output, labels)
                elif opt.loss_mode == 'MRAE':
                    loss = criterion_mrae(output, labels)
                elif opt.loss_mode == 'L1_MRAE':
                    # 权重可调，通常 L1 占主导，MRAE 辅助
                    loss = criterion_l1(output, labels) + 0.1 * criterion_mrae(output, labels)
            
            # 梯度累积逻辑 + 混合精度反向传播
            if opt.accumulation_steps > 1: 
                loss = loss / opt.accumulation_steps
                scaler.scale(loss).backward() 
                if ((iteration + 1) % opt.accumulation_steps) == 0:
                    scale = scaler.get_scale()
                    scaler.step(optimizer) 
                    scaler.update() 
                    # 若遇到梯度爆炸(NaN/Inf), scaler缩放会导致scale减小并跳过optimizer.step()
                    if scale <= scaler.get_scale():
                        scheduler.step() 
                    optimizer.zero_grad() 
            else:
                scaler.scale(loss).backward()
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scale <= scaler.get_scale():
                    scheduler.step()
                optimizer.zero_grad()
            
            losses.update(loss.data)
            iteration += 1
            
            # 更新进度条信息
            pbar.set_postfix({'Loss': f"{losses.avg:.6f}", 'lr': f"{lr:.8f}"})

            # [NTIRE 特有] 强制截断
            if opt.dataset == 'NTIRE' and i >= per_epoch_iteration - 1:
                break
        
        # 验证
        print("\n Validating...")
        mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss = validate(val_loader, model)
        print(" Result: MRAE {:.6f}, RMSE {:.6f}, PSNR {:.6f}, SAM {:.6f}, SSIM {:.6f}".format(mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss))
        
        # 保存模型
        if mrae_loss < record_mrae_loss:
            record_mrae_loss = mrae_loss
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            # 额外保存 best
            torch.save(model.state_dict(), os.path.join(opt.outf, 'model_best.pth'))
            print(f" Saved Best Model (MRAE: {mrae_loss:.6f})")
            
        if epoch % opt.save_epoch == 0:
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
        
        logger.info(" Epoch[%04d], lr: %.9f, Train Loss: %.9f, Val MRAE: %.6f, RMSE: %.6f, PSNR: %.4f, SAM: %.4f, SSIM: %.4f" % (
                    epoch, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss))
        
        if epoch >= opt.end_epoch:
            break
            
    return 0

# =============================================================================
# 4. 验证函数
# =============================================================================
def validate(val_loader, model):
    model.eval() 
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_ssim = AverageMeter()
    
    with torch.no_grad(): 
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # [NTIRE 特有] 裁剪高度 (因为NTIRE原图482x512，482不是32的倍数，网络下采样池化时会由于尺寸不匹配报错，所以裁剪到480)
            if opt.dataset == 'NTIRE':
                input = input[:,:,:480,:]
                target = target[:,:,:480,:]
            # [CAVE] 不裁剪，全图推理 (因为原图512x512，512本身是2的9次方，是极佳的尺寸，完全不会报错)

            # output = model(input)
            # output = torch.clamp(output, 0, 1)
            # 启用混合精度加速推理
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # 大图分块推理 (Patch-based Inference)
                # 因为像 AWAN/Transformer 这类带有全图 Attention 的模型，计算复杂度与分辨率呈平方关系
                # 512x512 的整图验证瞬间需要极高显存。我们将验证图拆成 128x128 的块来依次运算再无缝拼接
                b, c, h, w = input.shape
                ps = opt.patch_size # 采用和训练相同的 128x128 尺寸
                output = torch.zeros_like(target)
                
                for yi in range(0, h, ps):
                    for xi in range(0, w, ps):
                        y = yi
                        x = xi
                        # 如果到达边界不足以切一个完整的 patch，则向回退一点，保证进入模型的是标准的 128x128
                        if y + ps > h: y = max(0, h - ps)
                        if x + ps > w: x = max(0, w - ps)
                        
                        patch = input[:, :, y:y+ps, x:x+ps]
                        if opt.method == 'HPRN':
                            semantic_patch = generate_slic_semantic_labels(patch)
                            out_patch = model(patch, semantic_patch)
                        else:
                            out_patch = model(patch)
                        
                        # 填入整图对应的位置
                        output[:, :, y:y+ps, x:x+ps] = out_patch
            
            output = torch.clamp(output, 0, 1).float() # 将输出转回 float32 算损失防止溢出
            
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_sam = criterion_sam(output, target)
            loss_ssim = criterion_ssim(output, target)
            
            if not torch.isinf(loss_mrae) and not torch.isnan(loss_mrae): losses_mrae.update(loss_mrae.data)
            if not torch.isinf(loss_rmse) and not torch.isnan(loss_rmse): losses_rmse.update(loss_rmse.data)
            if not torch.isinf(loss_psnr) and not torch.isnan(loss_psnr): losses_psnr.update(loss_psnr.data)
            if not torch.isinf(loss_sam) and not torch.isnan(loss_sam): losses_sam.update(loss_sam.data)
            if not torch.isinf(loss_ssim) and not torch.isnan(loss_ssim): losses_ssim.update(loss_ssim.data)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_ssim.avg

if __name__ == '__main__':
    main()
