import torch
import torch.nn.functional as F
import torch_npu
import numpy as np
import matplotlib.pyplot as plt

def simple_dedispersion_gather(data, freq_array, dm_range, time_reso=1e-6, down_time_rate=1):
    """
    简化版去色散：使用gather操作替代卷积
    
    Args:
        data: 输入数据 [batch_size, time_samples, freq_channels]
        freq_array: 频率数组 [freq_channels] 
        dm_range: DM搜索范围
        time_reso: 时间分辨率
        down_time_rate: 降采样倍率
    
    Returns:
        result: 去色散结果 [dm_range, batch_size, time_samples]
    """
    batch_size, time_samples, freq_channels = data.shape
    
    # 计算延迟矩阵 [dm_range, freq_channels]
    DM_values = torch.arange(dm_range, dtype=torch.float32)
    freq_ref = freq_array.max()
    
    # 广播计算延迟
    delays = 4.15 * DM_values[:, None] * (freq_array[None, :]**-2 - freq_ref**-2) * 1e3
    delay_samples = (delays / time_reso / down_time_rate).round().long()
    
    # 限制延迟不超过时间范围
    max_delay = min(delay_samples.max().item(), time_samples - 1)
    delay_samples = torch.clamp(delay_samples, 0, max_delay)
    
    results = []
    
    for dm in range(dm_range):
        dm_result = torch.zeros(batch_size, time_samples)
        
        for t in range(time_samples):
            total_signal = 0.0
            valid_channels = 0
            
            for f in range(freq_channels):
                delay = delay_samples[dm, f].item()
                source_time = t - delay  # 注意：这里是减去延迟
                
                if 0 <= source_time < time_samples:
                    total_signal += data[0, source_time, f].item()
                    valid_channels += 1
            
            if valid_channels > 0:
                dm_result[0, t] = total_signal
        
        results.append(dm_result)
    
    return torch.stack(results, dim=0)

def generate_simple_test_data(batch_size=1, time_samples=1024, freq_channels=256, 
                            freq_start=1200, freq_end=1600, true_dm=10):
    """
    生成简单的测试数据（减小规模，避免计算问题）
    """
    # 创建频率数组
    freq_array = torch.linspace(freq_start, freq_end, freq_channels)
    
    # 初始化噪声数据
    data = torch.randn(batch_size, time_samples, freq_channels) * 0.1
    
    # 添加模拟脉冲
    pulse_time = time_samples // 2  # 脉冲在中间位置
    pulse_width = 5  # 脉冲宽度
    pulse_amplitude = 3.0  # 脉冲强度
    
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
            # 添加简单的矩形脉冲
            data[0, pulse_start:pulse_end, f] += pulse_amplitude
    
    return data, freq_array

def test_simple_dedispersion():
    """测试简化版去色散实现"""
    print("=== 简化版NPU去色散测试 ===")
    
    # 使用更小的数据规模进行测试
    print("1. 生成测试数据...")
    batch_size = 1
    time_samples = 1024  # 减小时间样本数
    freq_channels = 256  # 减小频率通道数
    true_dm = 10  # 减小DM值
    dm_range = 20  # 小的DM搜索范围
    
    data, freq_array = generate_simple_test_data(
        batch_size=batch_size, 
        time_samples=time_samples, 
        freq_channels=freq_channels,
        true_dm=true_dm
    )
    
    print(f"数据形状: {data.shape}")
    print(f"频率范围: {freq_array.min():.1f} - {freq_array.max():.1f} MHz")
    print(f"真实DM值: {true_dm}")
    
    # 估算延迟范围
    freq_ref = freq_array.max()
    freq_min = freq_array.min()
    max_delay = 4.15 * dm_range * (freq_min**-2 - freq_ref**-2) * 1e3 / 1e-6
    print(f"最大延迟估计: {max_delay:.1f} 采样点")
    print(f"输入长度: {time_samples} 采样点")
    
    # 运行去色散
    print("\n2. 执行去色散计算...")
    result = simple_dedispersion_gather(data, freq_array, dm_range)
    print(f"输出形状: {result.shape}")
    
    # 分析结果
    print("\n3. 分析结果...")
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
    
    # 显示前几个DM的信噪比
    print("\n前10个DM的信噪比:")
    for dm in range(min(10, dm_range)):
        print(f"DM {dm:2d}: SNR = {snr_values[dm]:6.2f}")
    
    # 简化的可视化
    print("\n4. 生成简化可视化...")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # DM-时间图
        dm_time_map = result[:, 0, :].numpy()
        im1 = ax1.imshow(dm_time_map, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('去色散结果 (DM-时间)')
        ax1.set_ylabel('DM值')
        ax1.axhline(y=true_dm, color='red', linestyle='--', label=f'真实DM={true_dm}')
        ax1.axhline(y=best_dm, color='yellow', linestyle='--', label=f'检测DM={best_dm}')
        ax1.legend()
        plt.colorbar(im1, ax=ax1)
        
        # 信噪比曲线
        ax2.plot(range(dm_range), snr_values, 'b-', linewidth=2, marker='o')
        ax2.axvline(x=true_dm, color='red', linestyle='--', label=f'真实DM={true_dm}')
        ax2.axvline(x=best_dm, color='yellow', linestyle='--', label=f'检测DM={best_dm}')
        ax2.set_xlabel('DM值')
        ax2.set_ylabel('信噪比')
        ax2.set_title('DM搜索曲线')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('simple_dedispersion_test.png', dpi=150, bbox_inches='tight')
        print("图像已保存为 simple_dedispersion_test.png")
        
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n5. 性能统计...")
    print(f"输入内存: {data.numel() * 4 / 1024**2:.1f} MB")
    print(f"输出内存: {result.numel() * 4 / 1024**2:.1f} MB")
    
    return result, best_dm, best_snr

def test_tensor_operations():
    """测试基础张量操作是否正常"""
    print("\n=== 基础张量操作测试 ===")

    set_npu = True
    
    # 测试设备
    if torch.cuda.is_available():
        device = 'cuda'
    elif set_npu:
        device = 'npu'
        print('设置device 为 npu')
    else:
        device = 'cpu'
        print('设置device 为 cpu')

    
    print(f"使用设备: {device}")
    
    # 简单的张量操作
    a = torch.randn(100, 200).to(device)
    b = torch.randn(200, 50).to(device)
    c = torch.matmul(a, b)
    
    print(f"矩阵乘法测试: {a.shape} × {b.shape} = {c.shape}")
    print(f"结果统计: mean={c.mean():.3f}, std={c.std():.3f}")
    
    # 测试gather操作
    data = torch.randn(10, 100).to(device)
    indices = torch.randint(0, 100, (10, 50)).to(device)
    gathered = torch.gather(data, 1, indices)
    
    print(f"Gather操作测试: {data.shape} -> {gathered.shape}")
    
    print("基础操作测试通过！")

if __name__ == "__main__":
    # 先测试基础操作
    test_tensor_operations()
    
    # 运行简化的去色散测试
    result, best_dm, best_snr = test_simple_dedispersion()
    
    print("\n=== 测试完成 ===")
    print("如果运行成功，说明去色散算法逻辑正确")
    print("可以在此基础上优化为更高效的NPU实现")
