import json
import numpy as np
from scipy.stats import pearsonr, kendalltau
import itertools
from scipy import stats


model_names = ['ResNet18','ResNet34']
source_datasets = ['CIFAR10', 'ImageNet']
train_types = ['Finetune', 'Retrain']
metrics = ['LEEP', 'GLEEP']
all_combinations = itertools.product(model_names, source_datasets, train_types, metrics)
for combination in all_combinations:
    model_name, Source_dataset, Train_type, metric = combination
    print(f"\n{'=' * 50}")
    print(f"Running experiment with: "
          f"Model={model_name}, Dataset={Source_dataset}, "
          f"Train={Train_type}, Metric={metric}")
    print(f"{'=' * 50}\n")
# 加载数据
    with open(f'result/{Train_type}/{model_name}/{Source_dataset}/metrics/{metric}.json', 'r') as f:
        leep_scores = json.load(f)

    with open(f'result/{Train_type}/{model_name}/{Source_dataset}/test_ACC/{metric}_ACC.json', 'r') as f:
        test_accuracies = json.load(f)

    # 提取数据并排序以确保一致性
    ids = sorted(leep_scores.keys())
    x = np.array([leep_scores[i] for i in ids])
    y = np.array([test_accuracies[i] for i in ids])
    std_x = np.std(x, ddof=1)  # 样本标准差
    std_y = np.std(y, ddof=1)
    # 计算 Pearson 相关系数
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    pearson_corr, pearson_p = pearsonr(x, y)
    calculated_slope = pearson_corr * (std_y / std_x)
    # 计算 Kendall 相关系数
    kendall_corr, kendall_p = kendalltau(x, y)
    print(f"Compare the relationship between {metric} score and test accuracy")
    print(f"Pearson 相关系数: {pearson_corr:.4f}, p-value: {pearson_p:.4f}")
    print(f"线性回归斜率: {slope:.4f}")
    print(f"Kendall 相关系数: {kendall_corr:.4f}, p-value: {kendall_p:.4f}")








