import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

wine = load_wine()
print(wine.data.shape)
print(wine.target.shape)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.2)
rfc = RandomForestClassifier(random_state=10)
rfc = rfc.fit(Xtrain, Ytrain)
score_r = rfc.score(Xtest, Ytest)
print("测试准确度{}".format(score_r))
rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)
dtc = DecisionTreeClassifier()
dtc_s = cross_val_score(dtc, wine.data, wine.target, cv=10)
# plt.plot(range(1,11),rfc_s,label="RandomForest")
# plt.plot(range(1,11),dtc_s,label="DecisionTree")
# plt.legend()
# plt.show()
superpa = []
for i in range(200):
    rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    superpa.append(rfc_s)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 201), superpa)
plt.show()
print("最高准确率为: {}-{}".format(superpa.index(max(superpa)), max(superpa)))
