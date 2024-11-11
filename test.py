# Inst necessary lib
# Note: u will need to inst these first using pip!   numpy pandas scikit-learn matplotlib seaborn
# NumPy: 科学计算基础库
# Pandas: 数据分析与处理
# Matplotlib: 基础可视化
# Seaborn: 统计数据可视化
# Scikit-learn: 机器学习工具

Data preprocessing is important
Feature scaling can improve model performance
Random forest is a robust classification algorithm
Exploratory data analysis (EDA) can help understand the data!






# 导入关键库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 加载鸢尾花(Iris Dataset)数据集
::IRIS_DATASET:: = load_iris()
::FEATURES:: = ::IRIS_DATASET::.data
::TARGETS:: = ::IRIS_DATASET::.target

# 数据探索
::IRIS_DATAFRAME:: = pd.DataFrame(
    data=np.c_[::FEATURES::, ::TARGETS::],
    columns=::IRIS_DATASET::.feature_names + ['target']
)

# 可视化数据分布
plt.figure(figsize=(10, 6))
sns.pairplot(::IRIS_DATAFRAME::, hue='target', palette='viridis')
plt.show()

# 数据预处理
::X_TRAIN::, ::X_TEST::, ::Y_TRAIN::, ::Y_TEST:: = train_test_split(
    ::FEATURES::, ::TARGETS::, test_size=0.3, random_state=42
)

::SCALER:: = StandardScaler()
::X_TRAIN_SCALED:: = ::SCALER::.fit_transform(::X_TRAIN::)
::X_TEST_SCALED:: = ::SCALER::.transform(::X_TEST::)

# 训练随机森林分类器
::ML_MODEL:: = RandomForestClassifier(n_estimators=100, random_state=42)
::ML_MODEL::.fit(::X_TRAIN_SCALED::, ::Y_TRAIN::)

# 模型预测与评估
::PREDICTIONS:: = ::ML_MODEL::.predict(::X_TEST_SCALED::)
print(classification_report(::Y_TEST::, ::PREDICTIONS::))
