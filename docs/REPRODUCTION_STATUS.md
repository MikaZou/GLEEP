# GLEEP 主实验复现状态

## 结论

当前状态是“主结果可部分复现，严格端到端复现尚未完成”。现有 JSON 足以恢复论文表1–3的主要相关性结果；新的独立实现已经可以审计、验表、从 logits 重算 GLEEP/LEEP、流式前向和逐任务训练，但本机完整 EXP2 数据与 CUDA 环境目前不可用。

## 已恢复的数值

EXP1 使用普通 Kendall tau-b 和 Pearson 重新计算得到：

| 方法 | Kendall 平均值 | Pearson 平均值 | 与论文主表 |
| --- | ---: | ---: | --- |
| LEEP | 0.544895 | 0.489779 | 一致，论文平均值存在截断 |
| NLEEP | 0.438537 | 0.522347 | 不一致 |
| LogME | 0.488056 | 0.444450 | 一致 |
| PACTran | 0.143937 | 0.000308 | 不一致 |
| SFDA | 0.597375 | 0.734311 | 一致 |
| GLEEP | 0.581168 | 0.615408 | 一致 |

论文表2写 GLEEP Pearson 为 `0.615`，正文写成 `0.618`。

EXP2 的八个 LEEP 设置平均 Pearson/Kendall 为 `0.888818/0.956669`，八个 GLEEP 设置为 `0.912186/0.950826`，与论文平均值相符。16 行中有 14 行在绝对误差不超过 `0.0015` 的口径下同时匹配；另外两行是 CIFAR10 Finetune LEEP 的 ResNet18/ResNet34 Kendall 值疑似对调。

## 数据与代码完整性

- 论文称 EXP1 有 11 个模型，但完整指标 JSON 每个数据集只有 10 个模型；InceptionV3 未参与当前表格计算。
- EXP2 的代码范围是 $k=2,\ldots,100$，因此通常为 99 个任务，而不是正文声称的 100 个。
- `Retrain/ResNet18/ImageNet/GLEEP` 仅有任务 `002` 至 `065`，共 64 对数据。
- `Retrain/ResNet34` 两种源数据的下游检查点均缺少全部 99 个任务，但历史 accuracy/metric JSON 存在。
- 原目录发现 11 个全 NUL Python 文件和 1 个语法错误文件；新实现不导入这些文件。
- 本地 EXP2 的 CIFAR100 `train`、`test`、`meta` 文件头均为全零，压缩包也不完整，不能用于真实数据 smoke。
- 官方代码中的 ResNet18/34 深度定义互换；CIFAR10 源分类头实际为 100 维；ImageNet LEEP 分支混用了 CIFAR10 分类头。`published` 模式保留这些行为，`corrected` 模式才修正。

## 本机验证

- 7 项单元测试全部通过。
- EXP1 Flowers 的 MobileNetV2、ResNet34 全量 LEEP 与历史结果绝对误差分别约为 `9.7e-10`、`2.1e-9`。
- 固定 GMM 随机状态后，两组 EXP1 GLEEP 子任务重复运行结果完全相同。
- EXP2 两个模型、两种源和端点任务的检查点加载、模型前向与指标链路已在 CPU synthetic smoke 中通过；synthetic 分数不作为论文结果。
- RTX 3050 Ti 可被 PyTorch 识别，但当前 PyTorch 1.13.1/CUDA 11.6/cuDNN 8.3 的最小卷积与完整模型探针失败。本机 GPU 结果不能作为代码失败依据，应在 `environment.yml` 的隔离环境或服务器上复测。

详细证据位于：

- `reports/audit.md`
- `reports/verification.md`
- `reports/smoke.md`
- `reports/package_manifest.json`

## 推荐执行顺序

```powershell
python -m gleep_repro audit
python -m gleep_repro verify --profile published
python -m gleep_repro smoke --device cuda
```

服务器完整重算前，先让正常 smoke 下载一份新的 CIFAR100 到 `.cache/data`，不要复用原目录中的全零文件。
