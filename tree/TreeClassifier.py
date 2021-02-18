import os

import graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
print(wine.data.shape)
print(wine.target.shape)
print(wine.feature_names)
print(wine.target_names)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.2)
classifier = tree.DecisionTreeClassifier(criterion="entropy", random_state=20, splitter="random", max_depth=3,
                                         min_samples_leaf=10, min_samples_split=10)
classifier = classifier.fit(Xtrain, Ytrain)
score = classifier.score(Xtest, Ytest)
print("预测准确度：{}".format(score))
feature_name = ['jiujing', 'pingguosuan', 'hui', 'huidejian', 'mei', 'zongfeng', 'leihuangtong', 'feihuangwanlei',
                'huaqinsu', 'yanseqinagdu', 'sediao', 'od280/od315', 'puansuan']
class_name = ["QIN", "XUELI", "BEIERMODE"]
dot_data = tree.export_graphviz(classifier, feature_names=feature_name, class_names=class_name, filled=True,
                                rounded=True)
graph = graphviz.Source(dot_data)
graph.view(directory=os.getcwd(), cleanup=True)
print(list(zip(feature_name, classifier.feature_importances_)))
