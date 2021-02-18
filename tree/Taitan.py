import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(os.getcwd() + os.sep + "data.csv")
data.drop(["Cabin", "Name", "Ticket"], inplace=True, axis=1)

data["Age"] = data["Age"].fillna(data["Age"].mean())
data = data.dropna(axis=0)
labels = data["Embarked"].unique().tolist()
# print(labels)
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
data["Sex"] = (data["Sex"] == "male").astype("int")
print(data.info())
print(data.head())
x = data.iloc[:, data.columns != "Survived"]
y = data.iloc[:, data.columns == "Survived"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2)
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

classifier = DecisionTreeClassifier(random_state=10)
classifier.fit(Xtrain, Ytrain)
score = classifier.score(Xtest, Ytest)
print(score)
score = cross_val_score(classifier, x, y, cv=10).mean()
print(score)
tr = []
te = []
for i in range(10):
    cla = DecisionTreeClassifier(random_state=10, max_depth=i + 1, criterion="entropy")
    cla = cla.fit(Xtrain, Ytrain)
    score_tr = cla.score(Xtrain, Ytrain)
    score_te = cross_val_score(cla, x, y, cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1, 11), tr, color="red", label="train")
plt.plot(range(1, 11), te, color="blue", label="test")
plt.xticks(range(1, 11))
plt.legend()
plt.show()
gini_threholds = np.linspace(0, 0.5, 20)
parameters = {"criterion": ("gini", "entropy")
    , "splitter": ("best", "random")
    , "max_depth": [*range(1, 10)]
    , "min_samples_leaf": [*range(1, 50, 5)]
    , "min_impurity_decrease": gini_threholds
              }
ifier = DecisionTreeClassifier(random_state=10)
gs = GridSearchCV(ifier, parameters, cv=10)
gs = gs.fit(Xtrain, Ytrain)
# gs.best_params_#从输入的参数和参数取值的列表中返回最佳组合
# gs.best_score_#网格搜索后的模型的评判标准
print(gs.best_params_)
print(gs.best_score_)
