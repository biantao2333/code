import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import torch
import torch.nn as nn
from torch.autograd import Variable

# ==========================================
# 绘图全局设置 (使其符合学术期刊的排版标准)
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6)
})

# ==========================================
# 工具函数 & 评价指标类
# ==========================================
def safe_load_mat(path):
    """ 安全读取 .mat 文件中的高光谱数据 cube """
    try:
        data = sio.loadmat(path)
        cube = data.get('cube', data.get('rad', None))
        return cube
    except Exception as e:
        with h5py.File(path, 'r') as f:
            cube = np.array(f['cube'])
            # 处理可能的维度转置 (如果形状为 [31, H, W])
            if cube.ndim == 3 and cube.shape[0] == 31: 
                cube = cube.transpose(1, 2, 0)
        return cube

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, im_fake, im_true):
        sum1 = torch.sum(im_true * im_fake, 1)
        sum2 = torch.sum(im_true * im_true, 1)
        sum3 = torch.sum(im_fake * im_fake, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        
        # 防止数值不稳定
        t = torch.clamp(t, -1.0, 1.0)
        
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
        SAM = sumangle * 180 / 3.14159256
        return SAM

criterion_sam = Loss_SAM()

# ==========================================
# 核心绘图功能
# ==========================================

# ---------------------------------------------------------
# 1. 绘制特定像素的光谱反射率曲线 (Spectral Reflectance Curve)
# ---------------------------------------------------------
def plot_spectral_curve(gt_path, pred_paths_dict, x, y, save_path):
    gt_cube = safe_load_mat(gt_path)
    if gt_cube is None: return
    
    bands = np.linspace(400, 700, 31)
    
    plt.figure()
    gt_spectrum = gt_cube[x, y, :]
    plt.plot(bands, gt_spectrum, 'k-', linewidth=3, label='Ground Truth')
    
    styles = ['r--', 'b-.', 'g:', 'm--', 'c-.']
    for (model_name, path), style in zip(pred_paths_dict.items(), styles):
        pred_cube = safe_load_mat(path)
        pred_spectrum = pred_cube[x, y, :]
        plt.plot(bands, pred_spectrum, style, linewidth=2, label=model_name)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'Spectral Curve at Pixel ({x}, {y})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 成功保存光谱曲线图至: {save_path}")

# ---------------------------------------------------------
# 2. 绘制参数量-性能权衡散点图 (Complexity vs. Accuracy)
# ---------------------------------------------------------
def plot_tradeoff_scatter(save_path):
    models = ['HSCNN+', 'AWAN', 'MST++', 'GMSR', 'MambaSSR']
    
    # 示例数据（需替换真实结果）
    params = [3.2, 4.5, 2.3, 1.8, 1.1]     
    psnrs = [32.1, 33.5, 34.2, 33.8, 34.5] 
    
    plt.figure()
    colors = ['gray', 'orange', 'purple', 'green', 'red']
    
    for i, model in enumerate(models):
        marker = '*' if model == 'MambaSSR' else 'o'
        size = 300 if model == 'MambaSSR' else 150
        plt.scatter(params[i], psnrs[i], s=size, c=colors[i], marker=marker, label=model, edgecolors='black')
        plt.text(params[i]+0.1, psnrs[i], model, verticalalignment='center')
        
    plt.xlabel('Parameters (M)')
    plt.ylabel('PSNR (dB)')
    plt.title('Accuracy vs. Parameters Trade-off')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 成功保存 Trade-off 散点图至: {save_path}")

# ---------------------------------------------------------
# 3. 波段级别指标曲线图 (Band-wise Performance)
# ---------------------------------------------------------
def plot_bandwise_metrics(gt_path, pred_paths_dict, save_path, metric='RMSE'):
    gt_cube = safe_load_mat(gt_path)
    bands_num = 31
    bands = np.linspace(400, 700, bands_num)
    
    plt.figure()
    styles = ['r-o', 'b-s', 'g-^', 'm-d', 'c-x']
    
    for (model_name, path), style in zip(pred_paths_dict.items(), styles):
        pred_cube = safe_load_mat(path)
        metrics_list = []
        
        for b in range(bands_num):
            gt_b = gt_cube[:, :, b]
            pred_b = pred_cube[:, :, b]
            
            if metric == 'RMSE':
                val = np.sqrt(np.mean((gt_b - pred_b)**2))
            elif metric == 'PSNR':
                mse = np.mean((gt_b - pred_b)**2)
                val = 10 * np.log10(1.0 / (mse + 1e-8))
            metrics_list.append(val)
            
        plt.plot(bands, metrics_list, style, linewidth=2, markersize=6, label=model_name)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(f'{metric}')
    plt.title(f'Band-wise {metric}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 成功保存波段 {metric} 曲线图至: {save_path}")

# ---------------------------------------------------------
# 4. MRAE 误差热力图 (MRAE Error Heatmap)
# ---------------------------------------------------------
def save_mrae_heatmap(gt_path, pred_path, save_path):
    gt_cube = safe_load_mat(gt_path)
    pred_cube = safe_load_mat(pred_path)
    
    abs_error = np.abs(pred_cube - gt_cube)
    mrae_map = np.zeros(gt_cube.shape[:2])
    
    for i in range(gt_cube.shape[0]):
        for j in range(gt_cube.shape[1]):
            valid_bands = gt_cube[i, j, :] > 0.01
            if np.sum(valid_bands) > 0:
                mrae_map[i, j] = np.mean(abs_error[i, j, valid_bands] / gt_cube[i, j, valid_bands])
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mrae_map, cmap='jet', vmin=0, vmax=0.15) 
    plt.axis('off')
    cbar = plt.colorbar(shrink=0.8)
    cbar.ax.set_ylabel('MRAE Error', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print(f"✅ 成功保存 MRAE 误差热力图至: {save_path}")

# ---------------------------------------------------------
# 5. SAM 误差伪彩图 (Integrated from sam_show.py)
# ---------------------------------------------------------
def save_sam_heatmap(gt_path, pred_path, save_path, vmax=15):
    """ 计算两个高光谱图像的 SAM (光谱角) 并将其保存为喷墨伪彩图 """
    gt_cube = safe_load_mat(gt_path)
    pred_cube = safe_load_mat(pred_path)
    
    # 转换为 torch.Tensor 用于 Loss_SAM 计算, Shape 需处理为 [1, 31, H, W]
    GT = Variable(torch.from_numpy(gt_cube).float()).permute(2, 0, 1).unsqueeze(0)
    OUTPUT = Variable(torch.from_numpy(pred_cube).float()).permute(2, 0, 1).unsqueeze(0)
    
    # 计算 SAM (不进行反向传播，不需要 cuda)
    with torch.no_grad():
        sam = criterion_sam(OUTPUT, GT) # 输出形状应为 [1, H, W]
    
    sam_map = sam[0].numpy()
    
    plt.imsave(save_path, sam_map, cmap='jet', vmin=0, vmax=vmax)
    print(f"✅ 成功保存 SAM 误差彩图至: {save_path}")


if __name__ == '__main__':
    # 模拟创建输出目录
    out_dir = './paper_figures'
    os.makedirs(out_dir, exist_ok=True)
    
    print("---------------------------------------------------------")
    print("🌟 论文分析图表生成工具 (集成了指标图与 SAM/MRAE 热力图) 🌟")
    print("---------------------------------------------------------")
    
    # 示例执行 Tradeoff 散点图
    plot_tradeoff_scatter(os.path.join(out_dir, 'tradeoff_curve.png'))
    
    print("代码骨架已就绪！后续在跑完 test.py 得到 .mat 输出后：")
    print("1. 调用相应的绘图函数 (如 plot_spectral_curve, save_sam_heatmap)。")
    print("2. 传入对应的 GT 和 推理预测的 `.mat` 即可自动生成可用于论文的高清图片。")
