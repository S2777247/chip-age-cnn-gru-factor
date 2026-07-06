# Chip-Age CNN-GRU Factor Reproduction

本项目参考华泰证券金工研报《基于筹码分层结构的端到端AI因子》，围绕筹码龄分层结构进行方法复现。项目实现了从筹码分布递推、人工筹码龄特征构造、RankIC 检验、分组回测，到 CNN-GRU 滚动训练与 IC Loss 因子评价的完整流程。

## 项目结构

- `src/get_csi1000_data.py`：获取中证1000样本股票基础行情数据
- `src/build_csi1000_chip_age_factor.py`：构造多股票筹码龄因子样本
- `src/build_cnn_gru_tensor_n300.py`：构造 CNN-GRU 输入张量
- `src/train_cnn_gru_n300_rolling_ic.py`：CNN-GRU 滚动训练与 IC Loss
- `src/run_csi1000_rankic.py`：人工特征 RankIC 检验
- `src/run_csi1000_group_backtest.py`：人工特征分组回测
- `figures/`：主要实验结果图

## 实验设置

- 数据来源：Tushare Pro
- 股票池：中证1000前300只股票
- 样本区间：2020-01-01 至 2026-04-23
- 输入结构：30 × 4 × 32
- 模型结构：CNN-GRU
- 优化器：AdamW
- 损失函数：IC Loss
- 评价指标：RankIC、RankIC t值、RankIC为正比例、分组收益

## 说明

由于数据权限和文件体积限制，本仓库不包含原始行情数据、Tushare Token、完整中间张量和模型权重文件。仓库仅用于展示项目代码结构、核心实现流程和实验结果图。
