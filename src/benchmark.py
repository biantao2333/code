import torch
import time
import numpy as np
from thop import profile
from thop import clever_format

import sys
import os
import warnings

# 忽略类似 thop 的 UserWarning 等无关紧要的警告信息
warnings.filterwarnings('ignore')

# 确保能相对导入上一级的 models 和同级的 model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import model_generator

def benchmark_models():
    # 这里列出你实现了的所有模型名称
    model_names = ['HSCNN+', 'AWAN', 'MST++', 'HPRN', 'GMSR', 'MambaSSR']
    
    # 模拟真实输入，NTIRE 可设为 480，CAVE 设为 512，由于 thop 在 512 的时候可能会遇到很深的递归算力瓶颈导致卡死
    # 学术界做 FLOPs 对比往往会使用单 patch 比如 (1, 3, 128, 128) 或者 (1, 3, 256, 256) 来比较
    # 或者对于需要高显存的 MST++，512x512算作测试时间可能极大。
    input_shape = (1, 3, 128, 128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {device} | Input shape: {input_shape}\n")
    print("-" * 100)
    print(f"{'Model Name':<15} | {'Params (M)':<12} | {'MACs/FLOPs (G)':<15} | {'Inf. Time (ms)':<15} | {'Peak Mem (MB)':<15}")
    print("-" * 100)

    for name in model_names:
        print(f"[{name}] Starting...", flush=True) # 进度提示
        try:
            # 清理显存以获得纯净的初始状态
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = model_generator(name).to(device)
            model.eval()
            
            dummy_input = torch.randn(input_shape).to(device)

            # 针对 HPRN 等需要额外输入的特殊处理
            if name == 'HPRN':
                # 构造符合 HPRN 需要的 dummy semantic_label, n_scales 默认为 4，尺寸为 HxW
                # 根据 HPRN.py 源码的 assert 或使用习惯，为了数值稳定和 embedding 选择，通常建议保持跟网络内部数据类型一致。
                # 由于 HPRN 源码中 semantic_labels 直接输入做了 float 操作，这里我们提供原汁原味的 float32
                dummy_label = torch.randint(0, 100, (1, 4, input_shape[2], input_shape[3])).float().to(device)
                inputs_tuple = (dummy_input, dummy_label)
            else:
                inputs_tuple = (dummy_input,)

            # =========================
            # 1. 计算 Params 和 FLOPs (MACs)
            # =========================
            import sys
            import os
            # 屏蔽 thop.profile 内部可能会产生的琐碎打印
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            macs, params = profile(model, inputs=inputs_tuple, verbose=False)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            
            # 使用 clever_format 自动转化为易读的 M 和 G 单位
            macs_str, params_str = clever_format([macs, params], "%.3f")

            # =========================
            # 2. 计算精准的推理速度 (Inference Time) & 峰值显存 (Peak Memory)
            # =========================
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            repetitions = 50
            timings = np.zeros((repetitions, 1))

            # a. GPU 预热 (消除第一次运行的代码缓存和显存加载开销)
            with torch.no_grad():
                for _ in range(10): # 增加预热次数到 10 次
                    _ = model(*inputs_tuple)

            # b. 使用 CUDA Event 真正精确计时
            with torch.no_grad():
                for rep in range(repetitions):
                    starter.record()
                    _ = model(*inputs_tuple)
                    ender.record()
                    # 强制同步等待执行完毕
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time

            avg_time = np.mean(timings) # 已经是毫秒(ms)
            
            # 获取计算此模型时分配的峰值显存
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # 转换为 MB

            # 打印表格行
            print(f"{name:<15} | {params_str:<12} | {macs_str:<15} | {avg_time:>10.2f} ms   | {peak_memory:>10.2f} MB")
            
            # 清理当前模型占用的显存，以防下一个模型报 OOM
            del model
            del dummy_input
            if name == 'HPRN':
                del dummy_label
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{name:<15} | Error: {str(e)}")

    print("-" * 80)

if __name__ == '__main__':
    benchmark_models()
