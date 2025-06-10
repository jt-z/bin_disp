# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# project information
project(acl_execute_disp)
# Compile options
add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")
set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()
set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
# 添加OpAPI路径
set(OPAPI_PATH "${INC_PATH}/opp/built-in/op_impl/ai_core/tbe/op_api")
set(LIB_PATH $ENV{NPU_HOST_LIB})
# 使用运行时库而非stub库
if (NOT DEFINED ENV{NPU_HOST_LIB})
    # 使用实际的运行时库，避免stub库的依赖问题
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/")
    set(LIB_PATH1 "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib/linux/aarch64/")
    message(STATUS "set runtime LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()
# Header path
include_directories(
    # ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ../inc
    ${CUST_PKG_PATH}/include
    ${OPAPI_PATH}/include
)
# add host lib path
link_directories(
    ${LIB_PATH}
    ${LIB_PATH1}
    ${CUST_PKG_PATH}/lib
    ${OPAPI_PATH}/lib/linux/aarch64
)
add_executable(execute_disp_op
    operator_desc.cpp
    op_runner.cpp
    main.cpp
    common.cpp
)
target_link_libraries(execute_disp_op
    opapi
    nnopbase
    ascendcl
    cust_opapi
    acl_op_compiler
    stdc++
    pthread
    rt
    dl
)
install(TARGETS execute_disp_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
