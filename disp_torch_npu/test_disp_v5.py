import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
import time

def gpu_dedispersion_v1(data, freq_array, dm_range, time_reso=1e-6, down_time_rate=1):
    """
    GPU优化版本1: 使用高级索引和广播
    """
    device = data.device
    batch_size, time_samples, freq_channels = data.shape
    
    # 计算延迟矩阵 [dm_range, freq_channels]
    DM_values = torch.arange(dm_range, dtype=torch.float32, device=device)
    freq_ref = freq_array.max()
    
    delays = 4.15 * DM_values[:, None] * (freq_array[None, :]**-2 - freq_ref**-2) * 1e3
    delay_samples = (delays / time_reso / down_time_rate).round().long()
    delay_samples = torch.clamp(delay_samples, 0, time_samples - 1)
    
    # 创建时间索引网格 [dm_range, freq_channels, time_samples]
    time_indices = torch.arange(time_samples, device=device)[None, None, :]  # [1, 1, T]
    source_times = time_indices - delay_samples[:, :, None]  # [DM, F, T]
    
    # 限制索引范围
    source_times = torch.clamp(source_times, 0, time_samples - 1)
    
    # 创建频率索引 [dm_range, freq_channels, time_samples]
    freq_indices = torch.arange(freq_channels, device=device)[None, :, None]
    freq_indices = freq_indices.expand(dm_range, -1, time_samples)
    
    # 使用gather进行批量索引 [dm_range, freq_channels, time_samples]
    gathered_data = data[0, source_times, freq_indices]
    
    # 在频率维度求和 [dm_range, time_samples]
    result = gathered_data.sum(dim=1)
    
    return result

def gpu_dedispersion_v2(data, freq_array, dm_range, time_reso=1e-6, down_time_rate=1):
    """
    GPU优化版本2: 矩阵乘法实现 (最快)
    """
    device = data.device
    batch_size, time_samples, freq_channels = data.shape
    
    # 计算延迟矩阵
    DM_values = torch.arange(dm_range, dtype=torch.float32, device=device)
    freq_ref = freq_array.max()
    
    delays = 4.15 * DM_values[:, None] * (freq_array[None, :]**-2 - freq_ref**-2) * 1e3
    delay_samples = (delays / time_reso / down_time_rate).round().long()
    delay_samples = torch.clamp(delay_samples, 0, time_samples - 1)
    
    # 创建采样矩阵 [dm_range * time_samples, time_samples * freq_channels]
    total_elements = time_samples * freq_channels
    sampling_matrix = torch.zeros(dm_range * time_samples, total_elements, device=device)
    
    # 填充采样矩阵
    for dm in range(dm_range):
        dm_offset = dm * time_samples
        for t in range(time_samples):
            row_idx = dm_offset + t
            for f in range(freq_channels):
                delay = delay_samples[dm, f].item()
                source_time = max(0, min(t - delay, time_samples - 1))
                col_idx = source_time * freq_channels + f
                sampling_matrix[row_idx, col_idx] = 1.0
    
    # 矩阵乘法: [dm_range * time_samples, 1]
    data_flat = data[0].flatten()  # [time_samples * freq_channels]
    result_flat = torch.matmul(sampling_matrix, data_flat)
    
    # 重塑结果 [dm_range, time_samples]
    result = result_flat.reshape(dm_range, time_samples)
    
    return result

def gpu_dedispersion_v3_optimized(data, freq_array, dm_range, time_reso=1e-6, down_time_rate=1):
    """
    GPU优化版本3: 分块处理 + 向量化 (推荐)
    """
    device = data.device
    batch_size, time_samples, freq_channels = data.shape
    
    # 计算延迟矩阵
    DM_values = torch.arange(dm_range, dtype=torch.float32, device=device)
    freq_ref = freq_array.max()
    
    delays = 4.15 * DM_values[:, None] * (freq_array[None, :]**-2 - freq_ref**-2) * 1e3
    delay_samples = (delays / time_reso / down_time_rate).round().long()
    delay_samples = torch.clamp(delay_samples, 0, time_samples - 1)
    
    # 分块处理，避免内存爆炸
    block_size = 128  # 每次处理128个时间点
    results = []
    
    for t_start in range(0, time_samples, block_size):
        t_end = min(t_start + block_size, time_samples)
        block_len = t_end - t_start
        
        # 当前块的时间索引 [block_len]
        time_block = torch.arange(t_start, t_end, device=device)
        
        # 计算源时间索引 [dm_range, freq_channels, block_len]
        source_times = time_block[None, None, :] - delay_samples[:, :, None]
        source_times = torch.clamp(source_times, 0, time_samples - 1)
        
        # 向量化索引
        # 创建批次索引
        batch_idx = torch.zeros_like(source_times)  # 都是0，因为batch_size=1
        
        # 重塑为一维索引
        flat_source = source_times.flatten()
        flat_freq = torch.arange(freq_channels, device=device).repeat(dm_range * block_len)
        flat_batch = batch_idx.flatten()
        
        # 使用advanced indexing
        gathered = data[flat_batch, flat_source, flat_freq]
        
        # 重塑并求和 [dm_range, freq_channels, block_len] -> [dm_range, block_len]
        gathered = gathered.reshape(dm_range, freq_channels, block_len)
        block_result = gathered.sum(dim=1)
        
        results.append(block_result)
    
    # 拼接所有块
    result = torch.cat(results, dim=1)
    
    return result

def benchmark_gpu_methods(data, freq_array, dm_range=20):
    """性能对比测试"""
    print("=== GPU加速方法性能对比 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'npu'
    data = data.to(device)
    freq_array = freq_array.to(device)
    
    print(f"设备: {device}")
    print(f"数据大小: {data.shape}")
    print(f"DM范围: {dm_range}")
    
    methods = [
        ("CPU版本(参考)", None),  # 原始版本
        ("GPU高级索引", gpu_dedispersion_v1),
        ("GPU矩阵乘法", gpu_dedispersion_v2), 
        ("GPU分块优化", gpu_dedispersion_v3_optimized)
    ]
    
    results = {}
    
    for name, method in methods:
        if method is None:
            continue  # 跳过CPU版本
            
        print(f"\n测试 {name}...")
        
        # 预热
        if device == 'cuda':
            torch.cuda.synchronize()
        
        try:
            start_time = time.time()
            result = method(data, freq_array, dm_range)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"  执行时间: {duration:.4f} 秒")
            print(f"  输出shape: {result.shape}")
            print(f"  内存占用: {result.numel() * 4 / 1024**2:.1f} MB")
            
            # 计算吞吐量
            total_ops = dm_range * data.shape[1] * data.shape[2]
            throughput = total_ops / duration / 1e6
            print(f"  吞吐量: {throughput:.1f} M ops/sec")
            
            results[name] = {
                'time': duration,
                'result': result,
                'throughput': throughput
            }
            
        except Exception as e:
            print(f"  错误: {e}")
            results[name] = None
    
    # 验证结果一致性
    print("\n=== 结果验证 ===")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        result_list = list(valid_results.values())
        reference = result_list[0]['result']
        
        for name, data_dict in valid_results.items():
            result = data_dict['result']
            max_diff = torch.max(torch.abs(result - reference)).item()
            print(f"{name}: 最大差异 = {max_diff:.6f}")
    
    return results

def test_gpu_acceleration():
    """测试GPU加速效果"""
    print("=== GPU去色散加速测试 ===")
    
    # 生成测试数据
    batch_size = 1
    time_samples = 2048
    freq_channels = 512
    true_dm = 15
    dm_range = 32
    
    freq_array = torch.linspace(1200, 1600, freq_channels)
    data = torch.randn(batch_size, time_samples, freq_channels) * 0.1
    
    # 添加模拟脉冲
    pulse_time = time_samples // 2
    pulse_width = 8
    pulse_amplitude = 5.0
    
    freq_ref = freq_array.max()
    delays = 4.15 * true_dm * (freq_array**-2 - freq_ref**-2) * 1e3 / 1e-6
    delay_samples = delays.round().long()
    
    for f in range(freq_channels):
        delay = delay_samples[f].item()
        pulse_start = pulse_time + delay
        pulse_end = pulse_start + pulse_width
        
        if 0 <= pulse_start < time_samples and pulse_end <= time_samples:
            data[0, pulse_start:pulse_end, f] += pulse_amplitude
    
    print(f"数据: {data.shape}")
    print(f"真实DM: {true_dm}")
    
    # 运行性能测试
    results = benchmark_gpu_methods(data, freq_array, dm_range)
    
    # 分析最佳结果
    if results:
        best_method = min(results.keys(), key=lambda k: results[k]['time'] if results[k] else float('inf'))
        print(f"\n最快方法: {best_method}")
        
        if results[best_method]:
            result = results[best_method]['result']
            
            # 找到最佳DM
            snr_values = []
            for dm in range(dm_range):
                signal = result[dm, :]
                snr = signal.max() / signal.std() if signal.std() > 0 else 0
                snr_values.append(snr.item())
            
            best_dm = np.argmax(snr_values)
            print(f"检测到的DM: {best_dm} (误差: {abs(best_dm - true_dm)})")

if __name__ == "__main__":
    test_gpu_acceleration()
