#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import threading


def gen_golden_data_simple():
    time_reso = 1.0
    down_time_rate = 2.0
    freq = 1.0
    DM = 1.0
    y = 1.0

    input_x = (np.random.uniform(1, 100, [512, 1]).astype(np.float32))**(-2) # 输入数据的shape为 一维向量 512,1 的， 为512个观测频率波段。
    golden = np.zeros(512).astype(np.float32)  #  np.zeros()函数默认返回 float64 双精度的值, 需要转换为和输出结果数据类型一致的 np.float32。
    
    for j in range(100):
        for i in range(512):
            x = input_x[i, 0]
            golden[i] = 4.15 * DM * (x - freq**(-2)) * 1e3 / time_reso / down_time_rate + y
            
        # golden[i] = x - freq**(-2) # 检验算子部分add计算是否成功
        # golden[i] = 4.15 * DM * (x - freq**(-2)) * 1e3 / time_reso # 检验算子中间计算是否成功


    input_x.tofile("./input/inputfreq.bin")
    golden.tofile("./output/golden.bin")

def check_threads():
    print("当前线程数:", threading.active_count())  # 单线程输出1，多线程>1
    print("所有线程:", threading.enumerate())



if __name__ == "__main__":
    gen_golden_data_simple()
    check_threads()
