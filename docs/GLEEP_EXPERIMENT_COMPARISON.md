# GLEEP 实验结果与论文结果对照

更新时间：2026-09-05  
复现目录：`GLEEP_repro`  
复现实验最初在私有服务器工作区运行；公开报告只记录相对路径与产物哈希。

## 1. 总体结论

当前结果应表述为：**论文主表结果已实现分层复现，但尚未完成严格的端到端复现。**

| 复现层级 | 当前状态 | 结论 |
| --- | --- | --- |
| 从历史 JSON 重算表1–3 | 已完成 | EXP1 的4个主要指标匹配；EXP2 有14/16行匹配论文 |
| 从源模型重新计算迁移性分数 | 部分完成 | EXP2 的396个任务已全量重算；EXP1只完成抽样验证 |
| 从头重新训练下游模型 | 未完成 | EXP1和EXP2相关性仍使用历史 accuracy JSON |
| 从原始数据到论文全部图表 | 未完成 | 消融、时间实验和图4–5不在本轮范围内 |

“匹配”采用普通 Kendall tau-b 和 Pearson，容差为 $\pm 0.0015$。后续表格中 `P` 表示 Pearson，`K` 表示 Kendall tau-b。论文代码使用平均概率版 legacy LEEP，而不是 canonical log-LEEP。

## 2. 两类结果不能混为一谈

本文档同时报告两种复现口径：

1. **历史结果复算**：读取原实验目录中的 metric JSON 和 accuracy JSON，重新计算 Pearson/Kendall。这能判断现有实验文件是否可恢复论文表格，但不是独立重跑。
2. **固定种子独立重算**：从源模型、真实 CIFAR100 数据重新计算 LEEP/GLEEP，再与历史 accuracy JSON 关联。GLEEP 使用 `random_state=0`，因此可重复，但不一定逐项等于论文未固定随机状态的历史 GMM 结果。

## 3. EXP1：论文表1和表2

### 3.1 历史 JSON 复算结果

EXP1 使用11个目标数据集，每个完整指标文件实际包含10个模型。论文正文声称有11个模型，但 InceptionV3 没有进入现有表格计算。

| 指标 | 重算 Kendall | 论文 Kendall | 重算 Pearson | 论文 Pearson | 对照结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| LEEP | 0.544895 | 0.544 | 0.489779 | 0.489 | 匹配 |
| NLEEP | 0.438537 | 0.488 | 0.522347 | 0.531 | 不匹配 |
| LogME | 0.488056 | 0.488 | 0.444450 | 0.444 | 匹配 |
| PACTran | 0.143937 | 0.172 | 0.000308 | 0.045 | 不匹配 |
| SFDA | 0.597375 | 0.597 | 0.734311 | 0.734 | 匹配 |
| GLEEP | 0.581168 | 0.581 | 0.615408 | 0.615 | 匹配 |

论文表2中的 GLEEP 平均 Pearson 为 `0.615`，正文写成 `0.618`，两处存在内部矛盾。现有 JSON 重算结果 `0.615408` 支持表格中的 `0.615`。

### 3.2 独立重算进度

| 数据集 | 模型 | 当前 LEEP | 历史 LEEP | 绝对误差 | GLEEP 检查 |
| --- | --- | ---: | ---: | ---: | --- |
| Flowers | MobileNetV2 | 0.063496690927 | 0.063496689960 | $9.67\times10^{-10}$ | 真实子集重复两次完全一致 |
| Flowers | ResNet34 | 0.059276656599 | 0.059276654498 | $2.10\times10^{-9}$ | 真实子集重复两次完全一致 |

EXP1 尚未完成以下内容：

- 10个模型与11个数据集的110组完整流式前向和 GLEEP/LEEP 重算；
- NLEEP、LogME、PACTran、SFDA 的独立指标重算；
- 产生 EXP1 Finetune accuracy 的下游训练。

因此，EXP1 当前是“主表可恢复、局部指标链路已验证”，不是端到端复现。

## 4. EXP2：论文表3

### 4.1 实验设置

- 模型：ResNet18、ResNet34；
- 源数据：CIFAR10、ImageNet；
- 下游分支：Finetune、论文所称的 Retrain；
- 指标：LEEP、GLEEP；
- 目标任务：CIFAR100 固定类别排列的前 $k$ 类，$k=2,\ldots,100$；
- 实际任务数：99，而不是论文文字所称的100；
- 固定种子：类别选择 `seed=42`，GMM `random_state=0`；
- GMM：full covariance；
- 公式：legacy 平均概率版 LEEP；
- 代码语义：`published`，保留官方实现中的历史行为。

每个模型—源数据组合的迁移性分数只计算一次，然后分别与 Finetune 和 Retrain accuracy 关联。4个模型—源数据组合共完成 $4\times99=396$ 个真实评分任务。

### 4.2 LEEP：论文、历史复算和固定种子重算

| 分支 | 模型 | 源数据 | 配对数 | 论文 P | 历史 P | 重算 P | 论文 K | 历史 K | 重算 K | 固定种子对照 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Finetune | ResNet18 | CIFAR10 | 99 | 0.925 | 0.925463 | 0.925461 | 0.936 | 0.933210 | 0.933210 | Kendall 差异 |
| Finetune | ResNet18 | ImageNet | 99 | 0.793 | 0.792511 | 0.792511 | 0.943 | 0.943105 | 0.943105 | 匹配 |
| Finetune | ResNet34 | CIFAR10 | 99 | 0.902 | 0.902184 | 0.902188 | 0.933 | 0.936289 | 0.936289 | Kendall 差异 |
| Finetune | ResNet34 | ImageNet | 99 | 0.790 | 0.789970 | 0.789975 | 0.946 | 0.946403 | 0.946403 | 匹配 |
| Retrain | ResNet18 | CIFAR10 | 99 | 0.944 | 0.944350 | 0.944349 | 0.980 | 0.980210 | 0.980210 | 匹配 |
| Retrain | ResNet18 | ImageNet | 99 | 0.922 | 0.921976 | 0.921976 | 0.965 | 0.964956 | 0.964956 | 匹配 |
| Retrain | ResNet34 | CIFAR10 | 99 | 0.936 | 0.935519 | 0.935522 | 0.977 | 0.977631 | 0.977631 | 匹配 |
| Retrain | ResNet34 | ImageNet | 99 | 0.898 | 0.898568 | 0.898572 | 0.971 | 0.971552 | 0.971552 | 匹配 |

两项 CIFAR10 Finetune LEEP 的 Kendall 值疑似在论文 ResNet18/ResNet34 行之间互换：重算的 `0.933210` 更接近论文 ResNet34 行的 `0.933`，重算的 `0.936289` 更接近论文 ResNet18 行的 `0.936`。

### 4.3 GLEEP：论文、历史复算和固定种子重算

| 分支 | 模型 | 源数据 | 配对数 | 论文 P | 历史 P | 重算 P | 论文 K | 历史 K | 重算 K | 固定种子对照 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Finetune | ResNet18 | CIFAR10 | 99 | 0.955 | 0.954588 | 0.954829 | 0.931 | 0.930324 | 0.934859 | Kendall 差异 |
| Finetune | ResNet18 | ImageNet | 99 | 0.807 | 0.807393 | 0.811117 | 0.939 | 0.938982 | 0.944341 | 差异 |
| Finetune | ResNet34 | CIFAR10 | 99 | 0.928 | 0.927811 | 0.926614 | 0.930 | 0.931340 | 0.926392 | Kendall 差异 |
| Finetune | ResNet34 | ImageNet | 99 | 0.809 | 0.809555 | 0.797251 | 0.947 | 0.947227 | 0.941868 | 差异 |
| Retrain | ResNet18 | CIFAR10 | 99 | 0.970 | 0.970002 | 0.969964 | 0.976 | 0.976087 | 0.971965 | Kendall 差异 |
| Retrain | ResNet18 | ImageNet | 64 | 0.959 | 0.959456 | 0.959956 | 0.940 | 0.940476 | 0.944444 | Kendall 差异；仅64对 |
| Retrain | ResNet34 | CIFAR10 | 99 | 0.955 | 0.955321 | 0.955675 | 0.973 | 0.972683 | 0.972271 | 匹配 |
| Retrain | ResNet34 | ImageNet | 99 | 0.913 | 0.913361 | 0.903918 | 0.969 | 0.969491 | 0.965368 | 差异 |

历史 GLEEP JSON 来自未记录 GMM 随机状态的运行；固定 `random_state=0` 后，结果具有确定性，但部分相关系数不会逐项等于历史值。这里的“差异”不表示作业失败，而表示确定性重算没有落入论文三位小数的 $\pm0.0015$ 容差。

### 4.4 表3平均结果

| 指标 | 论文平均 P | 历史复算 P | 固定种子重算 P | 论文平均 K | 历史复算 K | 固定种子重算 K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEEP | 0.888 | 0.888818 | 0.888819 | 0.956 | 0.956669 | 0.956669 |
| GLEEP | 0.912 | 0.912186 | 0.909916 | 0.951 | 0.950826 | 0.950188 |

固定种子重算仍保留论文的总体趋势：

- GLEEP 的平均 Pearson 比 LEEP 高约 `0.02110`；
- GLEEP 的平均 Kendall 比 LEEP 低约 `0.00648`；
- 即 GLEEP 在拟合线性相关性方面更好，但排序相关性并未超过 LEEP。

按逐行 $\pm0.0015$ 容差统计：

- 历史 JSON 复算：14/16行同时匹配论文 Pearson 和 Kendall；
- 固定种子独立评分：7/16行同时匹配，其中 LEEP 为6/8，GLEEP 为1/8；
- 固定种子 GLEEP 的主要差异来自论文历史运行未保存 GMM 随机状态。

## 5. EXP2 中间缓存与后续方法优化

服务器已经保存4组完整中间输出：

```text
results/intermediate/exp2_logits/published/
├── ResNet18/CIFAR10/
├── ResNet18/ImageNet/
├── ResNet34/CIFAR10/
└── ResNet34/ImageNet/
```

每组包含：

- `labels.npy`；
- `features.npy`，分类头前512维特征；
- `clustering_logits.npy`，CIFAR10源为100维，ImageNet源为1000维；
- `prediction_logits.npy`，100维；
- `manifest.json`，记录形状、类型、SHA256、检查点哈希、预处理和99个任务类别集合。

缓存总大小约178 MiB。16个数组全部为有限值且 SHA256 与 manifest 一致。全部396个 LEEP 任务从缓存重算只需19秒；与逐任务前向结果相比，平均绝对误差为 $1.41\times10^{-7}$，最大绝对误差为 $1.71\times10^{-5}$。

后续修改 GMM、降维、协方差、聚类方式、概率映射或 GLEEP 评分公式时，可以直接读取缓存，不再执行源模型前向：

```bash
python -m gleep_repro run exp2 \
  --mode score \
  --score-input cache \
  --tasks 2:100 \
  --device cpu
```

## 6. 尚未复现的部分

1. EXP1 全部110个模型—数据集组合的独立评分；
2. EXP1 的 NLEEP、LogME、PACTran、SFDA 指标重新生成；
3. EXP1 下游 Finetune accuracy 重新训练；
4. EXP2 的 $2\times2\times2\times99=792$ 次下游 Finetune/“Retrain”训练；
5. `Retrain/ResNet18/ImageNet/GLEEP` 缺失的35条历史 accuracy，使该行只能计算64对；
6. 消融实验、时间实验和图4–5。

此外，论文中的“Retrain”代码只训练分类头、冻结骨干网络，实际属于 linear probing，而不是通常意义上的从头训练。

## 7. 最终复现判断

| 判断问题 | 回答 |
| --- | --- |
| 能否从现有文件恢复论文主要相关性表格？ | 可以，EXP1四个主要指标匹配，EXP2历史结果14/16行匹配 |
| 是否重新计算了 EXP2 的 GLEEP/LEEP？ | 是，4组、396个真实任务已完成 |
| 新的确定性 GLEEP 是否逐项等于论文？ | 否，平均趋势一致，但只有1/8个 GLEEP 设置同时匹配两个相关系数 |
| 是否重新训练了产生 accuracy 的下游模型？ | 否，当前仍复用历史 accuracy JSON |
| 是否可以宣称完整端到端复现？ | 不能 |
| 当前最准确的表述 | 已完成论文主实验的分层复现和 EXP2 迁移性评分全量复现 |

## 8. 证据文件

- 历史表格核验：[`reports/verification.md`](reports/verification.md)
- 服务器运行与缓存状态：[`SERVER_REPRODUCTION_STATUS.md`](SERVER_REPRODUCTION_STATUS.md)
- 抽样验证：[`reports/smoke.md`](reports/smoke.md)
- 论文表格录入值：[`published/paper_tables.json`](published/paper_tables.json)
- EXP2 实现：[`gleep_repro/exp2.py`](gleep_repro/exp2.py)
