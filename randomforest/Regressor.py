import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # 填补缺失值
from sklearn.model_selection import cross_val_score

boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100, random_state=10)
cross_val_score(regressor, boston.data, boston.target, cv=10,
                scoring="neg_mean_squared_error")
# sklearn当中多有的模型评估指标(打分)列表
sorted(sklearn.metrics.SCORERS.keys())
print(boston.data.shape)
x_full, y_full = boston.data, boston.target
n_samples = x_full.shape[0]
n_features = x_full.shape[1]
rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
print(n_missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)
x_missing = x_full.copy()
y_missing = y_full.copy()
x_missing[missing_samples, missing_features] = np.nan
x_missing = pd.DataFrame(x_missing)
print(x_missing)

# 使用均值进行填补
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_missing_mean = imp_mean.fit_transform(x_missing)  # 训练fit+导出

# 使用0进行填补
imp_0 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
x_missing_0 = imp_0.fit_transform(x_missing)

# 算法填补
x_missing_reg = x_missing.copy()
#找出数据集中，缺失值从小到大排列的特征们的顺序,有了这些特征的索引
sortIndex=np.argsort(x_missing_reg.isnull().sum(axis=0)).values
#构建新的特征矩阵（没有被选中去填充的特征+原始的标签）和新标签（被选中去填充的特征）


