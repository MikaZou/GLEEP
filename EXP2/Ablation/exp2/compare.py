import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.serif'] = ['Times New Roman']
# 1. 准备实验数据（根据您提供的表格）
data = {
    'Algorithm': ['Fine-tune']*8 + ['Re-train']*8,
    'Source': ['CIFAR10', 'CIFAR10', 'ImageNet', 'ImageNet', 'CIFAR10', 'CIFAR10', 'ImageNet', 'ImageNet']*2,
    'Target': ['CIFAR100']*16,
    'Source model': ['ResNet18', 'ResNet34', 'ResNet18', 'ResNet34']*4,
    'Metric': ['Pearson', 'Pearson', 'Pearson', 'Pearson', 'kendall', 'kendall', 'kendall', 'kendall']*2,
    'LEEP score': [0.9255, 0.9022, 0.7925, 0.7900, 0.9363, 0.9332, 0.9431, 0.9464,
                   0.9444, 0.9355, 0.9220, 0.8986, 0.9802, 0.9776, 0.9650, 0.9716],
    'GLEEP score': [0.9546, 0.9278, 0.8074, 0.8096, 0.9313, 0.9303, 0.9390, 0.9472,
                    0.9700, 0.9553, 0.9595, 0.9134, 0.9761, 0.9727, 0.9405, 0.9695]
}

df = pd.DataFrame(data)
print('GLEEP score pearson')
print([df[df['Metric'] == 'Pearson']['LEEP score']])
# 2. 计算差异值
df['Difference'] = df['GLEEP score'] - df['LEEP score']
df['Relative Diff (%)'] = (df['Difference'] / df['LEEP score']) * 100

# 3. 可视化分析

plt.figure(figsize=(15, 12))



# 3.1 整体性能对比（条形图）
plt.figure(figsize=(10, 10))
avg_leep = [df[df['Metric'] == 'Pearson']['LEEP score'].mean(),
            df[df['Metric'] == 'kendall']['LEEP score'].mean()]
avg_gleep = [df[df['Metric'] == 'Pearson']['GLEEP score'].mean(),
             df[df['Metric'] == 'kendall']['GLEEP score'].mean()]

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, avg_leep, width, label='LEEP', color='#1f77b4')
plt.bar(x + width/2, avg_gleep, width, label='GLEEP', color='#ff7f0e')

plt.xticks(x, ['Pearson', 'Kendall'])
plt.ylabel('Average correlation')
plt.title('LEEP vs GLEEP ')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, v in enumerate(avg_leep):
    plt.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center')
for i, v in enumerate(avg_gleep):
    plt.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center')
plt.savefig('average.png', dpi=300)



# 3.2 不同指标下的差异分布（箱线图）
plt.figure(figsize=(8, 8))
print('pearson')
pearson_diff = df[df['Metric'] == 'Pearson']['Relative Diff (%)']
print(pearson_diff)
kendall_diff = df[df['Metric'] == 'kendall']['Relative Diff (%)']
print('kendall')
print(kendall_diff)
box = plt.boxplot([pearson_diff, kendall_diff],
                  patch_artist=True,
                  labels=['Pearson', 'Kendall'],
                  medianprops={'color': 'black', 'linewidth': 3})
plt.xticks(fontsize=24, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
# 设置箱线图颜色
colors = ['#2ca02c', '#d62728']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Relative Difference (%)',fontsize = 24, fontname='Times New Roman')
plt.xlabel('Correlation Coefficient',fontsize = 24, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加中位数标记
medians = [np.median(pearson_diff), np.median(kendall_diff)]
for i, median in enumerate(medians):
    plt.text(i+0.78, median-0.05, f'{median:.1f}%',
             ha='center', fontsize=20, color='black', fontname='Times New Roman')
plt.savefig('boxplot.pdf', dpi=300)

