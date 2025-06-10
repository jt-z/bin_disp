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
    down_time_rate = 2
    freq = 1.0
    DM = 1
    y = 1
    # DMnum = 1
    # ynum = 1
    input_x = np.random.uniform(1, 100, [512, 1]).astype(np.float32)
    # inputDM = np.arange(32) * 45 * 1e3
    # inputy = np.arange(64)
    golden = np.zeros(512).astype(np.float32)
    
    # for DM in range(DMnum):
        # for y in range(ynum):
        #     for i in range(512):
        #         x = input_x[i, 0]
        #         golden[i+DM*y] = 4.15 * DM * (x**-2 - freq**-2) * 1e3 / time_reso / down_time_rate + y

    # DM = DMnum
    # y = ynum
    for i in range(511):
        x = input_x[i, 0]
        golden[i] = 4.15 * DM * (x**-2 - freq**-2) * 1e3 / time_reso / down_time_rate + y


    input_x.tofile("./input/inputfreq.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
