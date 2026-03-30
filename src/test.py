import argparse, os
import torch
from torch.autograd import Variable
import time
import numpy as np
from torch.utils.data import DataLoader
import scipy.io as sio
# 引入 CAVE 数据集
from hsi_dataset import TrainDataset, ValidDataset, CAVEValidDataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM, Loss_SSIM

import sys
sys.path.append("..")
from model import model_generator

def saveCube(path, cube, bands=np.linspace(400,700,num=31), norm_factor=None):
    mat_dict = {'cube': cube, 'bands': bands}
    if norm_factor is not None:
        mat_dict['norm_factor'] = norm_factor
    sio.savemat(path, mat_dict)

# model_input
parser = argparse.ArgumentParser(description="DL4sSR Spectral Recovery Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
# 核心测试参数，只需要这两个
parser.add_argument("--model_path", required=True, type=str, help="Direct path to the `.pth` model file (Supports Tab completion)")
parser.add_argument("--dataset", type=str, default='CAVE', choices=['NTIRE', 'CAVE'], help="dataset name")
# 其他
parser.add_argument("--data_root", type=str, default='dataset/')

opt = parser.parse_args()
cuda = opt.cuda

# -------------- 从传入的路径中智能解析所需信息 --------------
# 输入示例: checkpoint/CAVE_GMSR_b4_L1_MRAE_lr0.0001/2026_03_25_01_35_16/model_best.pth
model_path = opt.model_path.replace('\\', '/')

# 1. 自动提取模型名称 (比如 GMSR)
#    截取外层文件夹名称：CAVE_GMSR_b4_L1_MRAE_lr0.0001
folder_name = model_path.split('/')[-3]
#    去掉类似 'CAVE_' 前缀，提取出 'GMSR'
prefix = opt.dataset + '_'
if folder_name.startswith(prefix):
    opt.model = folder_name[len(prefix):].split('_b')[0]
else:
    # 如果没按常规命名，则默认或容错处理
    opt.model = folder_name.split('_')[1] if '_' in folder_name else 'CanNet'
print(f"Auto-detected Model Architecture: ./{opt.model}")

# 2. 自动配置结果输出路径 (替换 'checkpoint' 为 'output')
outbase = os.path.dirname(model_path).replace('checkpoint', 'output')
print(f"Results will be saved to: ./{outbase}")
# ------------------------------------------------------------

print(f"Loading weights from {model_path} ...")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = model_generator(opt.model)
# 设置 weights_only=True 消除 PyTorch 安全隐患警告
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
# 兼容保存时包含 'state_dict' 键值的情况
if 'state_dict' in checkpoint:
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
else:
    model.load_state_dict(checkpoint)

model=model.cuda()
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

with torch.no_grad():
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    criterion_sam= Loss_SAM()
    criterion_ssim = Loss_SSIM()
    
    # 根据数据集选择加载器
    if opt.dataset == 'NTIRE':
        val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
    elif opt.dataset == 'CAVE':
        val_data = CAVEValidDataset(data_root=opt.data_root)
        
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0)
    
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_ssim = AverageMeter()
    timelist=[]
    
    print(f"Start testing on {opt.dataset} dataset...")
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        
        # 仅对 NTIRE 数据集进行硬编码裁剪
        if opt.dataset == 'NTIRE':
            input = input[:,:,:480,:]
            target = target[:,:,:480,:]
        
        # compute output
        starttime = time.time()
        output = model(input)
        torch.cuda.synchronize() # 强制同步，防止异步导致死锁看起来像别的地方卡住
        endtime = time.time()
        timelist.append(endtime-starttime)

        # 限制输出范围
        output = torch.clamp(output, 0, 1)
        out_temp = output.cpu()
        out = out_temp[0, :, :, :].permute(1, 2, 0).numpy().astype(np.float32)

        # 计算指标
        loss_mrae = criterion_mrae(output, target)
        loss_rmse = criterion_rmse(output, target)
        loss_psnr = criterion_psnr(output, target)
        loss_sam = criterion_sam(output, target)
        loss_ssim = criterion_ssim(output, target)

        outfile = outbase
        isExists = os.path.exists(outfile)
        if not isExists:
            os.makedirs(outfile)
        # 根据数据集调整保存文件名
        if opt.dataset == 'NTIRE':
            save_name = '/ARAD_1K_%d.mat' % (i + 901)
        elif opt.dataset == 'CAVE':
            # 直接从 dataset 的 scene_dirs 列表中获取原始文件夹名
            # 因为 DataLoader shuffle=False，所以索引 i 是对应的
            full_path = val_loader.dataset.scene_dirs[i]
            scene_name = os.path.basename(full_path)
            save_name = f'/{scene_name}.mat'
            
        saveCube('%s%s' %(outfile, save_name), out)
        
        # record loss
        if not torch.isinf(loss_mrae) and not torch.isnan(loss_mrae):
            losses_mrae.update(loss_mrae.item())
        if not torch.isinf(loss_rmse) and not torch.isnan(loss_rmse):
            losses_rmse.update(loss_rmse.item())
        if not torch.isinf(loss_psnr) and not torch.isnan(loss_psnr):
            losses_psnr.update(loss_psnr.item())
        if not torch.isinf(loss_sam) and not torch.isnan(loss_sam):
            losses_sam.update(loss_sam.item())
        if not torch.isinf(loss_ssim) and not torch.isnan(loss_ssim):
            losses_ssim.update(loss_ssim.item())
            
        # 优化后简洁的单行打印，避免每张图刷屏太多
        display_name = scene_name if opt.dataset == "CAVE" else f"Image {i+1}"
        print(f"[{i+1}/{len(val_loader)}] {display_name} \n"
              f"MRAE: {loss_mrae.item():.4f}, RMSE: {loss_rmse.item():.4f}, "
              f"PSNR: {loss_psnr.item():.2f}, SAM: {loss_sam.item():.2f}, SSIM: {loss_ssim.item():.4f}")
print("\n")
print(f"Total Time Used: {np.sum(timelist):.2f}s (Avg: {np.mean(timelist):.2f}s/img)")
print("Final Accuracy Metrics:")
print("    MRAE     RMSE      PSNR     SAM      SSIM")
print("  {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(
    losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_ssim.avg))
