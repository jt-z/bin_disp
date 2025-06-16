# 核心代码工程目录1：自定义算子开发工程目录——disp

描述： 为算子开发的代码工程， 包含算子核心计算逻辑，kernel代码等。

disp算子工程以及dispACL算子测试工程代码
目前存在CPU计算结果形状不符合预期的问题以及NPU计算结果为0的问题

自定义算子工程的算子开发的相关文档：包含 核函数实现、Host侧算子实现， 算子的编译部署 和 系统(System Test)测试  https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html#ZH-CN_TOPIC_0000001979603133__section1537973913498 

代码库基准代码: 可以参照 cann 8.0.RC2 版本的 示例 add_custom算子代码：内部包括算子的核心Process()函数代码逻辑  https://gitee.com/ascend/samples/blob/8.0.RC2/operator/AddCustomSample/FrameworkLaunch/AddCustom/op_kernel/add_custom.cpp

代码运行全流程命令如下：
（此版本为不输出详细log信息的情况，适用于已经调通的代码，用于调整计算逻辑）
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

（此版本用于打印输出log文档，排查代码报错情况，用于排查代码报错问题）
cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/Disp
./build.sh
cd build_out/
./custom_opp_ubuntu_aarch64.run
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
