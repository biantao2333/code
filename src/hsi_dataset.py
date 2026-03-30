from torch.utils.data import Dataset
import numpy as np
import random
import cv2
cv2.setNumThreads(0)  # 避免多进程 DataLoader 时因 OpenCV 预加载导致的死锁！
cv2.ocl.setUseOpenCL(False)
import h5py
import os
import glob
import torch

# =============================================================================
# 工具函数：RGB 模拟 (物理一致性)
# =============================================================================

def get_camera_sensitivity():
    """
    返回 31 个波段 (400nm - 700nm, step 10nm) 对应的相机光谱灵敏度曲线。
    这里使用常用的 Nikon D700 模拟数据，或者 CIE 1964 标准。
    Shape: (31, 3) -> (Band, RGB)
    """
    # 这是一个近似的 Nikon D700 光谱响应矩阵，广泛用于 SR 论文
    # 对应波长: 400, 410, ..., 700 (共31个)
    # 数据来源参考自 MST++ / TS-Net 等开源代码
    sensitivity = np.array([
        [0.005, 0.004, 0.049], [0.008, 0.013, 0.136], [0.011, 0.036, 0.308],
        [0.013, 0.076, 0.519], [0.014, 0.126, 0.686], [0.014, 0.176, 0.783],
        [0.014, 0.224, 0.814], [0.016, 0.268, 0.796], [0.020, 0.318, 0.733],
        [0.031, 0.388, 0.647], [0.052, 0.496, 0.546], [0.086, 0.636, 0.436],
        [0.134, 0.764, 0.326], [0.196, 0.844, 0.228], [0.274, 0.868, 0.152],
        [0.364, 0.836, 0.098], [0.460, 0.764, 0.064], [0.556, 0.664, 0.042],
        [0.642, 0.548, 0.028], [0.712, 0.428, 0.018], [0.760, 0.316, 0.012],
        [0.782, 0.220, 0.008], [0.784, 0.148, 0.006], [0.768, 0.096, 0.004],
        [0.736, 0.060, 0.002], [0.692, 0.036, 0.002], [0.636, 0.024, 0.000],
        [0.576, 0.016, 0.000], [0.512, 0.008, 0.000], [0.448, 0.004, 0.000],
        [0.388, 0.000, 0.000]
    ], dtype=np.float32)
    return sensitivity

def simulate_rgb(hyper, sensitivity):
    """
    根据 HSI 和 灵敏度曲线模拟 RGB
    hyper: (H, W, 31)
    sensitivity: (31, 3)
    return: (H, W, 3)
    """
    # 矩阵乘法: (H*W, 31) @ (31, 3) -> (H*W, 3)
    h, w, c = hyper.shape
    hyper_flat = hyper.reshape(-1, c)
    rgb_flat = np.dot(hyper_flat, sensitivity)
    rgb = rgb_flat.reshape(h, w, 3)
    
    # 归一化到 [0, 1]
    # 注意：模拟出来的 RGB 值可能会超过 1，需要根据实际情况处理
    # 通常做法是除以一个最大值或者直接 Clip，这里采用 Max-Normalize 保证不过曝
    rgb = rgb / (rgb.max() + 1e-8)
    return rgb

# =============================================================================
# 1. NTIRE 2022 数据集加载类 (保持不变)
# =============================================================================

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        """
        初始化训练数据集
        :param data_root: 数据集根目录
        :param crop_size: 训练时裁剪的小块大小 (例如 64x64)
        :param arg: 是否进行数据增强 (旋转、翻转)
        :param bgr2rgb: 是否将 OpenCV 读取的 BGR 转为 RGB
        :param stride: 切片时的步长，步长越小，重叠越多，数据量越大
        """
        self.crop_size = crop_size
        self.hypers = [] # 存储所有高光谱图像数据 (内存消耗大)
        self.bgrs = []   # 存储所有 RGB 图像数据
        self.arg = arg
        
        # NTIRE 2022 数据集的原始图像尺寸固定为 482x512
        h, w = 482, 512  
        self.stride = stride
        
        # 计算每张大图可以切出多少个小块 (Patch)
        # 逻辑：(图像宽 - 裁剪宽) // 步长 + 1
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        # 高光谱数据路径
        hyper_data_path = f'{data_root}/Train_Spec/'
        # RGB 数据路径
        bgr_data_path = f'{data_root}/Train_RGB/'

        # 读取训练列表 txt 文件
        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        
        # --- 核心加载循环：将所有图片读取到内存中 ---
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path: continue
            
            # 1. 读取高光谱数据 (.mat)
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            # 调整维度：原始可能为 [H, W, C] 或其他，这里统一转置
            hyper = np.transpose(hyper, [0, 2, 1]) 
            
            # 2. 读取 RGB 数据 (.jpg)
            bgr_path = bgr_data_path + bgr_list[i]
            bgr = cv2.imread(bgr_path)
            
            # 颜色空间转换 BGR -> RGB
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # 3. RGB 数据归一化
            bgr = np.float32(bgr)
            # Min-Max 归一化到 [0, 1]
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            # 维度变换: [H, W, C] -> [C, H, W] (PyTorch 格式)
            bgr = np.transpose(bgr, [2, 0, 1])  
            
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print("\r Ntire2022 scene {} is loaded.".format(i), end='')
            
        self.img_num = len(self.hypers)
        # 数据集总长度 = 图片数量 * 每张图的切片数
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        """
        数据增强函数
        :param rotTimes: 旋转次数 (0, 1, 2, 3) -> (0, 90, 180, 270度)
        :param vFlip: 是否垂直翻转
        :param hFlip: 是否水平翻转
        """
        # 随机旋转
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2)) # 在 H, W 维度旋转
        # 随机垂直翻转
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # 随机水平翻转
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        """
        获取一个训练样本 (Patch)
        """
        stride = self.stride
        crop_size = self.crop_size
        
        # --- 坐标映射逻辑 ---
        # 1. 确定是第几张大图
        img_idx = idx // self.patch_per_img # //是整数除法
        # 2. 确定是这张图里的第几个切片
        patch_idx = idx % self.patch_per_img
        # 3. 计算切片在图中的行列索引
        h_idx = patch_idx // self.patch_per_line
        w_idx = patch_idx % self.patch_per_line
        
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        
        # --- 切片操作 (Cropping) ---
        # 根据计算出的索引和步长，切出 crop_size 大小的块
        bgr = bgr[:, h_idx*stride : h_idx*stride+crop_size, w_idx*stride : w_idx*stride+crop_size]
        hyper = hyper[:, h_idx*stride : h_idx*stride+crop_size, w_idx*stride : w_idx*stride+crop_size]
        
        # --- 数据增强 ---
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip) # 数据增强
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
            
        # 返回连续内存数组，防止 PyTorch 报错
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img * self.img_num

class ValidDataset(Dataset):
    """
    验证集加载类：不进行切片，返回整张图像用于评估
    """
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        
        # 读取验证列表
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            
            bgr_path = bgr_data_path + bgr_list[i]
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print("\r Ntire2022 scene {} is loaded.".format(i), end='')

    def __getitem__(self, idx):
        # 直接返回整张图
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)


# =============================================================================
# 2. CAVE 数据集加载类 (CAVETrainDataset & CAVEValidDataset)
# 1. 随机 20/12 划分  2. 物理模拟 RGB
# =============================================================================

def get_cave_split(data_root):
    """
    统一管理 CAVE 数据集的划分，确保训练集和验证集互斥且固定。
    """
    # 自动处理 data_root 路径问题
    cave_dir = os.path.join(data_root, 'CAVE') if not data_root.endswith('CAVE') else data_root
    
    all_scenes = sorted([
        os.path.join(cave_dir, d) for d in os.listdir(cave_dir) 
        if os.path.isdir(os.path.join(cave_dir, d)) and not d.startswith('.')
    ])
    
    # 设定固定随机种子，保证每次运行划分一致
    random.seed(1234) 
    random.shuffle(all_scenes)
    
    # 24个训练，8个验证
    train_scenes = all_scenes[:24]
    valid_scenes = all_scenes[24:]
    
    return train_scenes, valid_scenes

class CAVETrainDataset(Dataset):
    def __init__(self, data_root, crop_size=128, arg=True, stride=8):
        """
        初始化 CAVE 训练集
        data_root: CAVE数据集的根目录，里面应该包含32个场景的子文件夹
        路径按需加载 + Random Crop
        """
        self.crop_size = crop_size
        self.arg = arg
        
        # 获取划分后的训练集列表
        self.scene_dirs, _ = get_cave_split(data_root)
        
        # 获取光谱灵敏度矩阵
        self.sensitivity = get_camera_sensitivity()
        
        print(f'Init CAVE Train dataset (24 scenes) from {data_root} (Lazy Loading)...')
        
        # 预加载 24 张原图到内存中，24张 512x512x31 的图只占不到 1GB 系统内存
        # 同时避免了每次训练迭代都去硬盘读 31 张 png 导致的极慢 IO 速度瓶颈
        # 我们把每个 Epoch 采样的次数定义为 1000 次（可灵活调整，以加快验证评测和记录保存的节奏）
        self.length = 1000
        self.images_bgr = []
        self.images_hyper = []
        for i, scene_path in enumerate(self.scene_dirs):
            bgr, hyper = self.load_image(scene_path)
            self.images_bgr.append(bgr)
            self.images_hyper.append(hyper)
            print(f"\r CAVE Train scene {i+1}/{len(self.scene_dirs)} loaded into RAM.", end='')
        print()

    def load_image(self, scene_path):
        """根据路径读取并生成单张图片的 HSI 和 RGB"""
        bands = []
        for b in range(1, 32):
            band_files = glob.glob(os.path.join(scene_path, f"*_{b:02d}.png"))
            if not band_files: band_files = glob.glob(os.path.join(scene_path, f"*{b}.png"))
            if band_files:
                img = cv2.imread(band_files[0], -1)
                if img is not None:
                    if img.ndim == 3: img = img[:, :, 0]
                    bands.append(img)
        
        hyper = np.stack(bands, axis=2).astype(np.float32) / 65535.0 
        rgb = simulate_rgb(hyper, self.sensitivity)
        
        # [H, W, C] -> [C, H, W]
        hyper = np.transpose(hyper, [2, 0, 1]) 
        rgb = np.transpose(rgb, [2, 0, 1])
        return rgb, hyper

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # 数据增强: 虽然有了 Random Crop，但旋转和翻转依然有必要，
        # 因为它们改变了物体的方向和光照的物理朝向，能让模型学到不同视角的特征。
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        # 从内存中随机选一张图，并完全随机取块 (Random Crop)
        scene_idx = random.randint(0, len(self.scene_dirs) - 1)
        
        # 1. 直接从内存获取这整张图 (极快)
        bgr = self.images_bgr[scene_idx]
        hyper = self.images_hyper[scene_idx]
        
        # 2. Random Crop：随机生成左上角坐标
        _, h, w = bgr.shape
        cy = random.randint(0, h - self.crop_size)
        cx = random.randint(0, w - self.crop_size)
        
        bgr = bgr[:, cy : cy + self.crop_size, cx : cx + self.crop_size]
        hyper = hyper[:, cy : cy + self.crop_size, cx : cx + self.crop_size]
        
        # 4. 旋转和翻转增强
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
            
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.length

class CAVEValidDataset(Dataset):
    """
    CAVE 验证集：加载所有图片，不切片
    """
    def __init__(self, data_root):
        self.hypers = []
        self.bgrs = []
        
        # 获取划分后的验证集列表
        _, self.scene_dirs = get_cave_split(data_root)
        
        # 获取光谱灵敏度矩阵
        self.sensitivity = get_camera_sensitivity()
        
        print(f'Loading CAVE Validation set (8 scenes) from {data_root}...')
        for i, scene_path in enumerate(self.scene_dirs):
            # 加载 HSI (31波段)
            bands = []
            for b in range(1, 32):
                band_files = glob.glob(os.path.join(scene_path, f"*_{b:02d}.png"))
                if not band_files: band_files = glob.glob(os.path.join(scene_path, f"*{b}.png"))
                if band_files:
                    # 确保读取时不带 Alpha 通道，或者处理 Alpha 通道
                    # cv2.imread(..., -1) 会读取所有通道。如果原图是 RGBA，就会读出 4 通道。
                    # 这里我们强制只取前两个维度（如果是灰度图）或者处理多通道问题
                    img = cv2.imread(band_files[0], -1)
                    if img is not None:
                        # 如果读出来是多通道 (H, W, C)，只取第一个通道，或者转灰度
                        if img.ndim == 3:
                            img = img[:, :, 0] 
                        bands.append(img)
            
            if len(bands) != 31: continue

            try:
                hyper = np.stack(bands, axis=2)
            except ValueError:
                continue
                
            hyper = hyper.astype(np.float32) / 65535.0
            
            # --- 物理模拟 RGB ---
            rgb = simulate_rgb(hyper, self.sensitivity)

            # 维度变换
            hyper = np.transpose(hyper, [2, 0, 1])
            rgb = np.transpose(rgb, [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(rgb)
            print(f"\r CAVE Valid scene {i+1}/8 loaded.", end='')
        print(f"\nTotal valid scenes loaded: {len(self.hypers)}")

    def __getitem__(self, idx):
        return np.ascontiguousarray(self.bgrs[idx]), np.ascontiguousarray(self.hypers[idx])

    def __len__(self):
        return len(self.hypers)
