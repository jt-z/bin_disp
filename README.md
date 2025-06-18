# 核心代码工程目录1：自定义算子开发工程目录——disp

描述： 为算子开发的代码工程， 包含算子核心计算逻辑，kernel代码等。

disp算子工程以及dispACL算子测试工程代码
目前存在CPU计算结果形状不符合预期的问题以及NPU计算结果为0的问题

自定义算子工程的算子开发的相关文档：包含 核函数实现、Host侧算子实现， 算子的编译部署 和 系统(System Test)测试  https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html#ZH-CN_TOPIC_0000001979603133__section1537973913498 

代码库基准代码: 可以参照 cann 8.0.RC2 版本的 示例 add_custom算子代码：内部包括算子的核心Process()函数代码逻辑  https://gitee.com/ascend/samples/blob/8.0.RC2/operator/AddCustomSample/FrameworkLaunch/AddCustom/op_kernel/add_custom.cpp

## 代码性能测试
### 手动测试
| | CPU      | NPU |
|----------- |----------- | ----------- |
|100 |2.8s      | 3.5s       |
|1000 |3s   | 4.5s        |
|10000 |22s   | 10s        |
|100000 |210s   | 65s        |

**性能分析**
![e0692bc5df3fa3ab53f2ad0290378be](https://github.com/user-attachments/assets/bfcf870c-2909-486f-8918-78f1ae7bddd9)

**CPU运行情况**

单线程cpu

![image](https://github.com/user-attachments/assets/d652570c-8684-4b51-b0c4-b56fbeb92b17)

计算的输入数据

(np.random.uniform(1, 100, [512, 1]).astype(np.float32))**(-2)

形状为（512，1），范围为（1,100）的，32位浮点数的，-2次方，的数组

计算公式的情况

golden[i] = 4.15 * DM * (x - freq**(-2)) * 1e3 / time_reso / down_time_rate + y

输出数据也是形状为（512,1）的数组

**npu使用情况**

![7a3a1a79083eb8a7b992dabb8c140f5](https://github.com/user-attachments/assets/e35d4d84-8e0a-431b-9145-0af5615beaf9)

算子运行时，NPU温度增加了两度，在温度上没有太大的波动。NPU 内存占用比空闲时略高（增加了约 45MB），使用了全部分配的 10 个大页内存页。显存占用99MB。

大致逻辑

算子接收的输入数据是已经计算好-2次幂的数据，这样可以绕过使用昇腾API计算次幂的问题，因为暂时没有找到合适API进行该操作。

整体计算分为四个步骤，首先计算两个freq值的差值，然后计算整个公式的乘法部分，然后分别计算两个除法，最后计算整体的加法部分。

Adds 计算x与freq**(-2)的差值得到结果1

Muls 计算结果1与4.15 * DM * 1e3的乘积得到结果2

Muls 计算结果2与time_reso的商得到结果3

Muls 计算结果3与down_time_rate的商得到结果4

Adds 计算结果4与y的和得到最终结果

## 比较分析下原有的整个disp函数中我们这部分大部分的计算占比

![image](https://github.com/user-attachments/assets/15fbc276-c83c-4858-a92b-29ce1e964d29)

从上图可以看出drafs函数中计算量比较大，时间复杂度比较高的部分是循环部分。我们将idx部分写成算子之后，最内层的循环还是需要进行其他操作，所以该函数的循环还是需要运行。所以后续需要考虑能否通过编写算子或者算子组合将多层循环消除，这样可以很大提高模型的性能。

## 项目的相关代码和数据

## 容器系统、 驱动版本、典型路径等环境

**容器系统**

![image](https://github.com/user-attachments/assets/76e9dab5-a6f9-4d29-a575-59e033d75ae4)

![image](https://github.com/user-attachments/assets/18572222-af53-421c-8fc8-ee201562a7c1)

**驱动版本**

![image](https://github.com/user-attachments/assets/2c3429cb-9890-4c98-b165-5348724d685c)

**典型路径**

部分常用头文件路径

/usr/local/Ascend/ascend-toolkit/8.0.RC2.2/aarch64-linux/include/acl/acl.h

/usr/local/Ascend/ascend-toolkit/8.0.RC2.2/aarch64-linux/include/acl/acl_op_compiler.h

/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp/build_out/autogen/aclnn_de_disp.h

## 代码运行全流程命令如下：
（此版本为不输出详细log信息的情况，适用于已经调通的代码，用于调整计算逻辑）
```
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp
./build.sh
cd build_out/
./custom_opp_ubuntu_aarch64.run
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp_gc
python3 scripts/gen_data.py
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp_gc/build
cmake ../src
make
cd ../output/
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1
./execute_disp_op
cd ..
python3 scripts/verify_result.py output/outputfreq.bin output/golden.bin
```

（此版本用于打印输出log文档，排查代码报错情况，用于排查代码报错问题）
```
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp
./build.sh
cd build_out/
./custom_opp_ubuntu_aarch64.run
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp_gc
python3 scripts/gen_data.py
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp_gc/build
cmake ../src
make
cd ../output/
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
./execute_disp_op > log.txt
cd ..
python3 scripts/gen_data.py
python3 scripts/verify_result.py output/outputfreq.bin output/golden.bin
```

## bug记录
### msopgen: command not found
官方教程是在sample目录下新建自定义算子目录，但是会遇到msopgen命令找不到的问题，如下图所示
![image](https://github.com/user-attachments/assets/3f271c3b-e68d-405a-9efa-e6405b414002)
在bin目录下进行msopgen操作显示JSON路径错误，但是JSON文件是存在的。在JSON文件的路径下进行msopgen操作显示msopgen命令不存在
**解决方法**
在bin目录下使用msopgen命令生成算子目录

### 两段式接口显示没有声明
算子测试工程两段式接口显示没有声明，具体情况如下：
https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0070.html
已经按照这个文档，进行了ACL算子测试工程的编写
https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0062.html
也已经按照这个文档进行了自定义算子的开发
自定义算子已经编译部署好了
但是在ACL算子测试工程中，运行make命令的时候，出现了以下报错

![image](https://github.com/user-attachments/assets/dbfe8bee-edc4-49a9-a4d6-32d3dc154a6a)

**问题原因**：没有包含头文件#include "aclnn_de_disp"

### libnnopbase.so 依赖两个符号，但它们在 Ascend 工具链的所有库中都不存在

![image](https://github.com/user-attachments/assets/ff7a0641-28db-4836-839f-659cbc826fbd)

主要是由于CMakeLists中的路径设置问题，重新修改CMakeLists中路径之后问题得到解决

### Get Operator Workspace failed. error code is 561003

![image](https://github.com/user-attachments/assets/4f97fa82-bdb1-47d5-be3f-3f4224ad3444)

![image](https://github.com/user-attachments/assets/90666fda-0a97-49bd-ba94-554954bded32)

算子只能处理float32类型的数据，但是在ACL算子测试工程中设置的数据类型是float16，所以出现了数据类型不匹配的问题

### 507015问题

![image](https://github.com/user-attachments/assets/6f469c3a-1795-48fe-90c6-c7440a034c8e)

一般是由于算子编写问题导致，需要到算子kernel侧代码中寻找问题

### CPU和NPU计算结果维度不匹配的问题

主要是因为循环中计算结果没有进行拼接的问题导致结果的覆盖

### CPU端计算结果维度问题

CPU端计算结果维度总是设定维度的二倍

由于numpy中zeros默认数据类型是float16，导致计算所得张量维度变为设定的二倍

### NPU端计算结果得0的问题

之前技术人员让注释掉算子process部分，导致算子一直不能被调用。现在已经取消了process的注释。算子可以被正常调到。但是算子不能完整的被执行。

针对算子kernel侧的代码进行了逐一的排查，发现将算子中buffer数目修改成1之后，算子可以正常被调用执行。主要原因是算子编写过程中没有按照双buffer来安排数据，但是buffer数目设置为了2，所以会导致执行失败。但是官方给出的样例中也没有显示设定双buffer的内容，具体问题还在研究中，代码已经跑通。

### 算子内部无法打印信息的问题

昇腾内部用于打印tensor信息的API，用可以成功使用其输出tensor形状。

AscendC::DumpTensor(freqLocal,5, this->tileLength);

### CPU和npu端接受到数据不一致的问题

由于npu端使用的是缓存的上次生成的数据而CPU使用的是新生成的数据，主要是由于算子加载在CPU计算之前，所以需要在算子加载数据之前就要将输入数据生成好，否则会出现NPU加载的是旧数据的问题，哪怕删除bin文件，算子中也依然保存着之前获取的数据。

### 算子中定值没有接受到的问题

由于官方文档和样例中没有太多的描述，所以先尝试在函数形参中给了默认值，发现算子内部无法获取到这部分的数值。因此先使用在算子内部设常数的方法确保算子计算结果正确。之后查阅文档的过程中发现了关于在tiling中赋初值的内容，尝试在tiling中进行值的设置，部分计算可以完成。

然后发现在除2操作时，会出现计算全部得0的问题，考虑到是由于精度问题，原始的函数设置的除2是整数，导致0.5直接截断为0，所以将一些相关的常数类型都改为了浮点数，之后计算通过。






