
# 核心代码工程目录2：测试 算子代码的正确性的代码工程目录——Disp_gc 

##  算子开发环节 —— ST系统测试环节

参照此文档： https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0070.html
文档为新版本文档，对比新旧版本代码库，发现差别不大。



后期的ACL代码编写，参照此文档 https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000022.html


## Bug和进度记录等

- [ ] 算子逻辑正确性验证：解决算子的计算结果为0的问题
    - [x] 实际上是因为算子未调用 Process()函数导致的。
    - [ ] de_disp.cpp 中要调用Process()函数，重新编译并进行 单算子API调用测试。
        - [ ] 目前遇到问题： 测试的时候，在  Synchronize stream 阶段报错
            - [ ] 错误信息： 根据AI的分析，怀疑可能是 workspace size 设置的问题。 

- [ ] 算子逻辑正确性验证：生成的测试数据的维度的问题
  - [ ] gen_data.py的代码中的数据生成




## 相关术语1： 算子开发环节 —— ST系统测试环节

算子的ST测试指的是**System Test（系统测试）**，这是CANN（Compute Architecture for Neural Networks）开发套件中用于验证自定义算子功能正确性的测试框架。

#### ST测试的核心作用

ST测试主要用于：
- **功能验证**：确保自定义算子在实际硬件环境中能正确执行
- **回归测试**：验证算子在不同输入条件下的稳定性
- **性能基准**：为算子性能优化提供基准数据

#### 关键组件解析

从你提供的文档来看，ST测试框架包含：

1. **msOpST工具**：核心测试执行引擎
2. **测试用例定义文件**：JSON格式，定义输入输出规格和测试参数
3. **测试报告**：st_report.json，包含详细的执行结果

#### 测试用例配置要点

JSON配置文件中的关键字段：
- `input_desc`/`output_desc`：定义张量的format、type、shape等属性
- `data_distribute`：数据分布策略（如uniform均匀分布）
- `value_range`：数值范围控制
- `soc_version`：指定目标昇腾AI处理器型号

这种设计允许开发者系统性地验证算子在**不同数据规模、数值范围和硬件配置下的行为**，是昇腾算子开发流程中的标准验证环节。

作为算法工程师，你应该特别关注测试用例的覆盖率设计，确保边界条件和异常情况都得到充分测试。