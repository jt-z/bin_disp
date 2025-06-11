import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def create_dispersion_kernel(dm_value, freq_array, time_reso=1e-6, down_time_rate=1, max_kernel_size=None):
    """
    为特定DM值创建色散卷积核
    
    Args:
        dm_value: 色散测度值
        freq_array: 频率数组 [freq_channels]
        time_reso: 时间分辨率 (秒)
        down_time_rate: 降采样倍率
        max_kernel_size: 最大核大小限制
    
    Returns:
        kernel: 卷积核 [freq_channels, 1, kernel_size]
    """
    freq_channels = len(freq_array)
    
    # 计算各频率相对最高频率的延迟（采样点数）
    freq_ref = freq_array.max()
    delays = 4.15 * dm_value * (freq_array**-2 - freq_ref**-2) * 1e3
    delay_samples = (delays / time_reso / down_time_rate).round().long()
    
    # 确定卷积核大小（需要覆盖最大延迟，但不能超过输入长度）
    max_delay = delay_samples.max().item()
    if max_kernel_size is not None:
        kernel_size = min(max(max_delay + 1, 3), max_kernel_size)
    else:
        kernel_size = max(max_delay + 1, 3)
    
    # 创建卷积核：每个频率通道有独立的核
    kernel = torch.zeros(freq_channels, 1, kernel_size)
    
    for f in range(freq_channels):
        delay = delay_samples[f].item()
        if 0 <= delay < kernel_size:
            kernel[f, 0, delay] = 1.0  # 在对应延迟位置设置权重
        # 如果延迟超出核大小，使用最后一个位置（近似处理）
        elif delay >= kernel_size:
            kernel[f, 0, kernel_size - 1] = 1.0
    
    return kernel

def npu_dedispersion_v3(data, freq_array, dm_range, time_reso=1e-6, down_time_rate=1):
    """
    方案3: 深度可分离卷积实现去色散
    
    Args:
        data: 输入数据 [batch_size, time_samples, freq_channels]
        freq_array: 频率数组 [freq_channels] 
        dm_range: DM搜索范围
        time_reso: 时间分辨率
        down_time_rate: 降采样倍率
    
    Returns:
        result: 去色散结果 [dm_range, batch_size, time_samples_out]
    """
    batch_size, time_samples, freq_channels = data.shape
    
    # 转换数据维度为卷积所需格式: [batch_size, freq_channels, time_samples]
    data = data.transpose(1, 2)
    
    # 限制最大卷积核大小为输入长度的一半，避免核过大
    max_kernel_size = time_samples // 2
    
    results = []
    for dm in range(dm_range):
        # 为当前DM创建卷积核，限制核大小
        kernel = create_dispersion_kernel(dm, freq_array, time_reso, down_time_rate, max_kernel_size)
        
        # 检查核大小是否合理
        kernel_size = kernel.shape[2]
        if kernel_size > time_samples:
            print(f"警告: DM={dm} 的核大小 {kernel_size} 超过输入长度 {time_samples}，跳过")
            # 创建全零结果
            dummy_result = torch.zeros(batch_size, max(time_samples - kernel_size + 1, 1))
            results.append(dummy_result)
            continue
        
        try:
            # 深度可分离卷积：每个频率通道独立卷积
            conv_result = F.conv1d(data, kernel, groups=freq_channels, padding=0)
            
            # 在频率维度求和：[batch_size, freq_channels, time_out] -> [batch_size, time_out]
            dedispersed = conv_result.sum(dim=1)
            results.append(dedispersed)
            
        except RuntimeError as e:
            print(f"DM={dm} 卷积失败: {e}")
            # 创建备用结果
            dummy_result = torch.zeros(batch_size, max(time_samples - kernel_size + 1, 1))
            results.append(dummy_result)
    
    # 找到最小的输出长度，统一所有结果的大小
    min_length = min(r.shape[1] for r in results)
    results_trimmed = [r[:, :min_length] for r in results]
    
    # 堆叠所有DM结果: [dm_range, batch_size, time_samples_out]
    return torch.stack(results_trimmed, dim=0)

def generate_test_data(batch_size=1, time_samples=2048, freq_channels=512, 
                      freq_start=1200, freq_end=1600, true_dm=30):
    """
    生成包含模拟脉冲的测试数据
    
    Args:
        batch_size: 批次大小
        time_samples: 时间采样点数
        freq_channels: 频率通道数
        freq_start: 起始频率 (MHz)
        freq_end: 结束频率 (MHz) 
        true_dm: 真实DM值
    
    Returns:
        data: 测试数据 [batch_size, time_samples, freq_channels]
        freq_array: 频率数组 [freq_channels]
    """
    # 创建频率数组
    freq_array = torch.linspace(freq_start, freq_end, freq_channels)
    
    # 初始化噪声数据
    data = torch.randn(batch_size, time_samples, freq_channels) * 0.1
    
    # 添加模拟脉冲
    pulse_time = time_samples // 2  # 脉冲在中间位置
    pulse_width = 10  # 脉冲宽度
    pulse_amplitude = 5.0  # 脉冲强度
    
    time_reso = 1e-6  # 1微秒时间分辨率
    down_time_rate = 1
    
    # 为每个频率计算色散延迟
    freq_ref = freq_array.max()
    delays = 4.15 * true_dm * (freq_array**-2 - freq_ref**-2) * 1e3
    delay_samples = (delays / time_reso / down_time_rate).round().long()
    
    # 在相应的延迟位置添加脉冲
    for f in range(freq_channels):
        delay = delay_samples[f].item()
        pulse_start = pulse_time + delay
        pulse_end = pulse_start + pulse_width
        
        if 0 <= pulse_start < time_samples and pulse_end <= time_samples:
            # 添加高斯形状的脉冲
            t_indices = torch.arange(pulse_start, pulse_end)
            pulse_profile = pulse_amplitude * torch.exp(-0.5 * ((t_indices - pulse_time - delay) / 3)**2)
            data[0, pulse_start:pulse_end, f] += pulse_profile
    
    return data, freq_array

def test_npu_dedispersion():
    """测试NPU去色散实现"""
    print("=== NPU去色散卷积测试 ===")
    
    # 生成测试数据
    print("1. 生成测试数据...")
    batch_size = 1
    time_samples = 2048 
    freq_channels = 512
    true_dm = 30
    dm_range = 32  # 减小DM搜索范围，避免核过大
    
    data, freq_array = generate_test_data(
        batch_size=batch_size, 
        time_samples=time_samples, 
        freq_channels=freq_channels,
        true_dm=true_dm
    )
    
    print(f"数据形状: {data.shape}")
    print(f"频率范围: {freq_array.min():.1f} - {freq_array.max():.1f} MHz")
    print(f"真实DM值: {true_dm}")
    
    # 预估最大延迟，调整搜索范围
    freq_ref = freq_array.max()
    freq_min = freq_array.min()
    max_delay_estimate = 4.15 * dm_range * (freq_min**-2 - freq_ref**-2) * 1e3 / 1e-6
    print(f"预估最大延迟: {max_delay_estimate:.1f} 采样点")
    print(f"输入长度: {time_samples} 采样点")
    
    if max_delay_estimate > time_samples // 2:
        new_dm_range = int(time_samples // 2 / (4.15 * (freq_min**-2 - freq_ref**-2) * 1e3 / 1e-6))
        print(f"调整DM搜索范围: {dm_range} -> {new_dm_range}")
        dm_range = new_dm_range
    
    # 运行去色散
    print("\n2. 执行去色散计算...")
    result = npu_dedispersion_v3(data, freq_array, dm_range)
    print(f"输出形状: {result.shape}")  # [dm_range, batch_size, time_samples_out]
    
    # 分析结果
    print("\n3. 分析结果...")
    # 计算每个DM的信噪比
    snr_values = []
    for dm in range(dm_range):
        signal = result[dm, 0, :]  # 取第一个batch
        noise_std = signal.std()
        signal_peak = signal.max()
        snr = signal_peak / noise_std if noise_std > 0 else 0
        snr_values.append(snr.item())
    
    best_dm = np.argmax(snr_values)
    best_snr = snr_values[best_dm]
    
    print(f"最佳DM: {best_dm} (真实值: {true_dm})")
    print(f"最佳信噪比: {best_snr:.2f}")
    print(f"DM误差: {abs(best_dm - true_dm)}")
    
    # 可视化结果
    print("\n4. 生成可视化...")
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 原始数据频谱图
        original_data = data[0].T  # [freq, time]
        ax1.imshow(original_data.numpy(), aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('原始数据 (频率-时间)')
        ax1.set_ylabel('频率通道')
        
        # DM-时间图
        dm_time_map = result[:, 0, :].numpy()  # [dm, time]
        ax2.imshow(dm_time_map, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('去色散结果 (DM-时间)')
        ax2.set_ylabel('DM值')
        if true_dm < dm_range:
            ax2.axhline(y=true_dm, color='red', linestyle='--', label=f'真实DM={true_dm}')
        ax2.axhline(y=best_dm, color='yellow', linestyle='--', label=f'检测DM={best_dm}')
        ax2.legend()
        
        # 信噪比曲线
        ax3.plot(range(dm_range), snr_values, 'b-', linewidth=2)
        if true_dm < dm_range:
            ax3.axvline(x=true_dm, color='red', linestyle='--', label=f'真实DM={true_dm}')
        ax3.axvline(x=best_dm, color='yellow', linestyle='--', label=f'检测DM={best_dm}')
        ax3.set_xlabel('DM值')
        ax3.set_ylabel('信噪比')
        ax3.set_title('DM搜索曲线')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('npu_dedispersion_test.png', dpi=150, bbox_inches='tight')
        print("图像已保存为 npu_dedispersion_test.png")
        # plt.show()  # 在服务器环境下注释掉
        
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n5. 性能统计...")
    print(f"内存占用估计: {data.numel() * 4 / 1024**2:.1f} MB (输入)")
    print(f"输出大小: {result.numel() * 4 / 1024**2:.1f} MB")
    
    return result, best_dm, best_snr

def benchmark_performance():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    import time
    
    test_configs = [
        {'time_samples': 1024, 'freq_channels': 256, 'dm_range': 32},
        {'time_samples': 2048, 'freq_channels': 512, 'dm_range': 64},
        {'time_samples': 4096, 'freq_channels': 1024, 'dm_range': 128},
    ]
    
    for config in test_configs:
        print(f"\n配置: {config}")
        
        # 生成数据
        data, freq_array = generate_test_data(
            time_samples=config['time_samples'],
            freq_channels=config['freq_channels']
        )
        
        # 计时
        start_time = time.time()
        result = npu_dedispersion_v3(data, freq_array, config['dm_range'])
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"执行时间: {duration:.3f} 秒")
        print(f"数据量: {data.numel() / 1e6:.1f}M 元素")
        print(f"吞吐量: {data.numel() / duration / 1e6:.1f}M 元素/秒")

if __name__ == "__main__":
    # 运行基本测试
    result, best_dm, best_snr = test_npu_dedispersion()
    
    # 运行性能测试
    benchmark_performance()
    
    print("\n=== 测试完成 ===")
    print("如果在NPU上运行，请确保:")
    print("1. 安装了NPU版本的PyTorch")
    print("2. 将device设置为NPU设备")
    print("3. 调用 .to('npu') 将数据移动到NPU")
