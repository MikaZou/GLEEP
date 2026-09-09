# UPR：方法、证明与适用边界

版本：2026-09-06，实验前固定的概念验证版本。目标是检验候选方向，不预设其优于GLEEP。

## 1. 定义与实现

输入为同一批无标签样本在多个源模型下的特征。评估模型 $m$ 时，参考池排除 $m$。将原始特征逐样本归一化，使用种子固定的高斯投影压缩到64维；附加常数1后得到 $F\in\mathbb R^{n\times65}$。所有65个系数，包括截距，统一正则化，$\alpha=0.01n/65$。

参考模型 $j$ 的投影特征为 $z_j(x)$。用采样的非零样本对距离中位数作为 $\sigma_j$，从 $\mathcal N(0,\sigma_j^{-2}I)$ 抽32个频率；拼接对应的cos/sin，再除以 $\sqrt{32}$，得到64维 $v_j(x)$。每行平方范数恰为1。距离采样与频率均只依赖无标签特征及固定随机种子。

令参考数量为 $p$，$V=p^{-1/2}[V_1,\ldots,V_p]$，$Q=VV^{\mathsf T}$。$Q$是连续的、半正定、对角元为1的关系近似，但不一定是真实同类概率；有限随机特征甚至可以产生负的非对角元。不把这些列叫作目标类别或概率。

定义：

$$
H=F(F^{\mathsf T}F+\alpha I)^{-1}F^{\mathsf T},\quad
D=\operatorname{diag}(1-H_{ii}),\quad R=D^{-1}(I-H).
$$

$$
\widehat{\mathcal E}=n^{-1}\|RV\|_F^2,
\qquad S_{\mathrm{UPR}}=-\widehat{\mathcal E}.
$$

计算 $HV=F(F^{\mathsf T}F+\alpha I)^{-1}(F^{\mathsf T}V)$，仅求解65维系统；由 $F$ 和其逆系统乘积计算杠杆值 $H_{ii}$。生产评分不构造 $n\times n$ 矩阵。若 $d$为设计维度，$r$为参考维度，主要计算量为 $O(nd^2+ndr+d^3)$，内存为 $O(n(d+r)+d^2)$，另计原始特征投影成本。

接口 `score_upr(projected_pool, candidate, seed)` 不接受标签、类别数、accuracy。基准适配层可以使用标签构造任务，理论评价层可使用标签计算oracle风险。

## 2. 命题一：固定设计的留一恒等式

令 $Y\in\mathbb R^{n\times K}$ 为任意响应矩阵（分类诊断时取one-hot）。将 $F,\alpha$固定，完整数据的岭解为 $B=(F^{\mathsf T}F+\alpha I)^{-1}F^{\mathsf T}Y$。删除第 $i$ 行后，仍用同一个 $\alpha$，得到 $B_{-i}$。

写 $G=F^{\mathsf T}F+\alpha I$，$f_i=F_i^{\mathsf T}$，$h_i=f_i^{\mathsf T}G^{-1}f_i$。删除样本后的Gram矩阵为 $G-f_if_i^{\mathsf T}$。Sherman–Morrison 恒等式给出

$$
(G-f_if_i^{\mathsf T})^{-1}
=G^{-1}+\frac{G^{-1}f_if_i^{\mathsf T}G^{-1}}{1-h_i}.
$$

乘以 $F^{\mathsf T}Y-f_iY_i$并整理，有

$$
B_{-i}=B-\frac{G^{-1}f_i(Y_i-f_i^{\mathsf T}B)}{1-h_i}.
$$

因此：

$$
Y_i-F_i B_{-i}=\frac{Y_i-(HY)_i}{1-H_{ii}}.
$$

由于 $\alpha>0$，$H$的特征值小于1，故 $H_{ii}<1$，分母为正。逐行平方求和即得：

$$
\mathcal E(Y)=\frac1n\sum_i\|Y_i-F_iB_{-i}\|_2^2
=\frac1n\|RY\|_F^2
=\frac1n\operatorname{tr}(RYY^{\mathsf T}R^{\mathsf T}).
$$

这里是固定特征和预处理后的留一拟合。任务带宽可使用全部无标签样本，但不能称为连预处理也在每次删除样本后重算的严格归纳交叉验证。

解析留一是经典线性平滑器性质，不是UPR原创贡献。相关基础：[Golub、Heath、Wahba (1979)](https://doi.org/10.1080/00401706.1979.10489751)。

## 3. 命题二：结构近似误差控制风险偏差

令 $A=YY^{\mathsf T}$，假设 $\|A-Q\|_2\le\varepsilon$。则

$$
|\mathcal E(Y)-\widehat{\mathcal E}|
=\frac1n|\operatorname{tr}((A-Q)R^{\mathsf T}R)|
\le\frac{\varepsilon}{n}\operatorname{tr}(R^{\mathsf T}R)
=:\delta.
$$

证明：$R^{\mathsf T}R$半正定，其核范数等于迹。使用谱范数与核范数的对偶不等式即可。此结论不要求 $Q$与标签矩阵独立，但没有给出如何无标签获得小 $\varepsilon$ 的保证。

两个模型可以具有不同参考矩阵和误差半径。若 $\widehat{\mathcal E}_a+\delta_a<\widehat{\mathcal E}_b-\delta_b$，则 $\mathcal E_a(Y)<\mathcal E_b(Y)$。这是平方风险排序，不是准确率排序。

## 4. 命题三：条件留一分类准确率下界

对 $p_i=F_iB_{-i}$取argmax分类，不要求 $p_i$归一化或非负。如果错分，则存在 $j\ne y_i$且 $p_{ij}\ge p_{iy_i}$。在这个半空间上，$\|e_{y_i}-p_i\|_2^2$的最小值为 $1/2$，在真实类与错误类坐标同时为 $1/2$时达到。因此

$$
\mathbf1\{\arg\max_jp_{ij}\ne y_i\}\le2\|e_{y_i}-p_i\|_2^2.
$$

逐样本平均，并使用命题二：

$$
\operatorname{ACC}_{\mathrm{LOO}}
\ge\max\{0,1-2\mathcal E(Y)\}
\ge\max\{0,1-2(\widehat{\mathcal E}+\delta)\}.
$$

这不是完整Finetune、交叉熵SGD linear probing或未来测试集风险的保证。若要得到总体风险界，需要额外的样本独立性、算法稳定性和分布假设。本次不宣称已完成这种推广。

## 5. 哪些结构假设足以控制 epsilon

考虑理想参考RBF核 $Q^*$。假设在参考模型 $j$ 下同类距离至多为 $a_j$，异类距离至少为 $b_j$。定义

$$
\eta_w=\frac1p\sum_j\left(1-e^{-a_j^2/(2\sigma_j^2)}\right),\quad
\eta_b=\frac1p\sum_j e^{-b_j^2/(2\sigma_j^2)}.
$$

若随机特征近似对所有非对角元误差不超过 $\xi$，对于大小为 $n_c$的类，其对应行的绝对误差和至多为

$$
(n_c-1)(\eta_w+\xi)+(n-n_c)(\eta_b+\xi).
$$

由于 $A-Q$对称，谱范数不超过最大绝对行和，于是该行和最大值给出一个 $\varepsilon$的充分上界。

这说明理论需要同类紧致、异类分离以及核近似足够准确。它也是警告：界会随 $n$放大，即使每个元素误差不大，最终准确率下界仍可能为0。没有语义条件时，不应声称模型共识自动满足该假设。

对于固定样本/带宽，cos/sin RFF的每个核元素是 $L=32$个独立cos变量的平均，各变量在 $[-1,1]$内且均值为理想RBF核。Hoeffding不等式给出单元素误差超过 $\xi$的概率至多 $2e^{-L\xi^2/2}$。对 $p$个参考模型及 $n(n-1)/2$个非对角无序样本对取union bound，失败概率至多 $p n(n-1)e^{-L\xi^2/2}$。因此概率至少 $1-\beta$时可取

$$
\xi=\min\left\{2,\sqrt{\frac{2}{L}\log\frac{p n(n-1)}{\beta}}\right\}.
$$

若上式截断为2，使用的是确定性的有界误差保证。仅32个频率时，保守界通常很宽。这是对随机频率抽样的理论概率陈述，不是给固定seed自动提供可用的小常数。本次使用oracle谱误差诊断界的有效性，不用未验证的小常数替代它。

## 6. 无标签不可辨识与反例

如果输入及所有源模型特征固定，但改变目标标签规则，无标签分数不会改变，监督可达性能却可能改变。这直接排除“无条件优于监督评分或普遍给出紧性能界”。

原型必须报告四类构造：共享语义、共享无关背景、独立噪声、常数表征。特别是所有模型都表示相同背景时，跨模型预测可能很容易，实际目标分类仍接近随机；常数表征也可能得到很高的UPR。这些是结构目标本身的失败，不应靠翻转分数方向或事后调参掩盖。

标签随机化后UPR严格不变，但oracle风险和准确率改变，这是信息边界检查。没有非零下界或者不能胜过简单无标签基线时，本版本应得出负结论。

常数表征反例中，严格平衡的二分类数据可能得到 **0而不是0.5的留一准确率**：删除当前样本后，另一类多一个训练样本，常数预测器会预测另一类，因此逐个留一全部错误。这与独立测试集上常数预测器约0.5的准确率不矛盾，不应把这一有限样本留一现象误当作实现故障。

## 7. 与GLEEP及已有工作的关系

GLEEP原文的式(14)是对数概率分数的下界，不是已证明的下确界；把目标标签换为簇标签也不能推出该界高于真实标签LEEP的界。实验legacy版本是平均概率，不能直接套用对数版证明。[LEEP](https://proceedings.mlr.press/v119/nguyen20b.html)

[RankMe](https://arxiv.org/abs/2210.02885)用有效秩无标签评价表征质量。UPR尝试估计跨模型关系的固定设计预测风险，必须实证胜过RankMe和普通核对齐才有继续价值。

[Task-relatedness (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/hash/d3602fc92fb8b9e0d55356c9e8815e2b-Abstract-Conference.html)已建立与任务相关性有关的迁移性分析，并讨论无目标标签的估计。因此无标签、有理论、使用参考任务本身都不是新的贡献。

UPR的留一恒等式、矩阵扰动界、平方损失到错分率的不等式均由标准工具得到。可能值得研究的部分是：能否可靠、廉价地估计目标同类关系，并得到有效预测风险。这一点必须由本轮结果决定；本次初步检索不构成“已证明文献中不存在相同方法”的声明。

## 8. 复现命令

```bash
python -m gleep_repro.upr_experiment \
  --output-dir results/upr-v1-20260906 --threads 1
```

默认种子0、1、2，99任务，4组缓存；种子0包含全部监督/无监督基线、消融和oracle诊断。种子1、2仅检查随机近似稳定性。输出已有时拒绝覆盖，修改代码后使用新的结果目录。

结果中的 `epsilon_oracle` 用真实标签计算，属于评价工具，不能作为可部署的无标签置信界。GMM计时固定在四组 $k=2,10,100$共12项；UPR重复3次取中位数，GMM每项1次。主相关性沿用先前396项固定种子GLEEP，明确其来自逐任务前向；缓存计时同时检查对应分数偏差。

计时复核可以保持所有分数不变，将新计时写入独立目录：

```bash
python -m gleep_repro.upr_experiment \
  --retime-report results/upr-v1-20260906/report.json \
  --output-dir results/upr-v1-audited-20260906 --threads 1
```

复核版在限制线程前加载sklearn，并在计时前执行小型合成GMM预热，避免把首次导入时间误算成评分开销；原始结果不覆盖。`score_origin`记录被复用结果的SHA256，`timing_protocol`独立记录新计时的代码哈希和线程设置。
