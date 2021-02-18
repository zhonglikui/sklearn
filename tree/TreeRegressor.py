from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

boston = load_boston()
regressor = tree.DecisionTreeRegressor(random_state=0)
cvs = cross_val_score(regressor, boston.data, boston.target, cv=10,
                      scoring="neg_mean_squared_error")  # 默认R平方，这里使用的是负的均方误差
print(cvs)
