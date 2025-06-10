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


def gen_golden_data_simple():
    time_reso = 1.0
    down_time_rate = 0.5
    freq = 1.0
    DMnum = 1
    ynum = 1
    input_x = np.random.uniform(1, 100, [512, 1]).astype(np.float32)
    # inputDM = np.arange(32) * 45 * 1e3
    # inputy = np.arange(64)
    golden = np.zeros(256)
    
    # for DM in range(DMnum):
        # for y in range(ynum):
        #     for i in range(512):
        #         x = input_x[i, 0]
        #         golden[i+DM*y] = 4.15 * DM * (x**-2 - freq**-2) * 1e3 / time_reso / down_time_rate + y

    DM = DMnum
    y = ynum
    for i in range(255):
        x = input_x[i, 0]
        golden[i] = 4.15 * DM * (x**-2 - freq**-2) * 1e3 / time_reso / down_time_rate + y


    # 生成随机数据（向量化）
    # DM_array = np.random.rand(131072, 1)       # (131072, 1)
    # y_array = np.random.rand(512, 1)            # (512, 1)
    # x_array = input_x[:, 0]                     # (512,) 取 input_x 的第一列

    # # 计算 (x^-2 - freq^-2)，假设 freq 是标量
    # term = (x_array**-2 - freq**-2)            # (512,)

    # # 广播计算 golden (131072, 512, 512)
    # golden = 4.15 * DM_array * term * 1e3 / time_reso / down_time_rate + y_array

    print("*****************************************************")

    input_x.tofile("./input/inputfreq.bin")
    # inputDM.tofile("./input/inputdm.bin")
    # inputy.tofile("./input/inputy.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
