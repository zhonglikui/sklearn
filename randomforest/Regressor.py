import sklearn
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100, random_state=10)
cross_val_score(regressor, boston.data, boston.target, cv=10,
                scoring="neg_mean_squared_error")
# sklearn当中多有的模型苹果指标(打分)列表
sorted(sklearn.metrics.SCORERS.keys())
