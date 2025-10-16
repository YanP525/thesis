import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ["LLaVA-NeXT", "Merged_0.25", "Merged_0.5", "Merged_0.75", "Merged_1.0", "DeepSeek-R1"]

# 数据集
datasets = ["RealWorldQA", "ai2d", "MME", "MMMU", "MathVista", "mmlu"]

# 准确率数据
accuracy = np.array([
    [59.48, 68.20, 68.15, 30.26, 28.50, 53.03],
    [0.00, 13.40, 0.30, 18.62, 2.50, 47.07],
    [31.11, 55.50, 51.61, 38.36, 24.90, 54.47],
    [35.56, 60.40, 59.68, 39.07, 20.70, 59.82],
    [26.93, 36.10, 48.39, 27.23, 15.60, 45.12],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 34.84]  # DeepSeek-R1 在部分数据集无结果
])

# 设置柱状图参数
x = np.arange(len(datasets))  # 数据集位置
width = 0.16   # 每个柱子宽度

fig, ax = plt.subplots(figsize=(20,6))


# 绘制每个模型的柱子

for i, model in enumerate(models):
    ax.bar(x + i*width, accuracy[i], width, label=model)

# 标签和标题
ax.set_xticks(x + width*2.5)  # 调整 x 轴刻度在柱子中间
ax.set_xticklabels(datasets)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy of Different Models Across Datasets")
ax.legend()
ax.set_ylim(0, 100)

# 在柱子上显示数值
for i in range(len(models)):
    for j in range(len(datasets)):
        if not np.isnan(accuracy[i][j]):
            ax.text(x[j] + i*width, accuracy[i][j]+1, f'{accuracy[i][j]:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
