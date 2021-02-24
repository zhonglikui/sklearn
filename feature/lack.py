import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

data=pd.read_csv(os.getcwd()+os.sep+"Narrativedata.csv",index_col=0)
#print(data.info())
print(data.head())
#填补AGE
Age=data.loc[:,"Age"].values.reshape(-1,1)#sklearn中特征矩阵必须是二位
print(Age[:10])
imp_mean=SimpleImputer()#实例化，默认均值填补
imp_median=SimpleImputer(strategy="median")#用中位数填补
imp_0=SimpleImputer(strategy="constant",fill_value=0)#用0填补
imp_mean=imp_mean.fit_transform(Age)
imp_median=imp_median.fit_transform(Age)
imp_0=imp_0.fit_transform(Age)
print("imp_mean:")
print(imp_mean[:10])
print("imp_median:")
print(imp_median[:10])
print("imp_0:")
print(imp_0[:10])
Embarked=data.loc[:,"Embarked"].values.reshape(-1,1)
print(Embarked[:20])
imp_mode=SimpleImputer(strategy="most_frequent")
#data.loc[:"Embarked"]=imp_mode.fit_transform(Embarked)

#使用filena在DataFrame里面进行填补
data2=pd.read_csv(os.getcwd()+os.sep+"Narrativedata.csv",index_col=0)
data2.loc[:"Age"]=data2.loc[:"Age"].fillna(data2.loc[:,"Age"].median())
data2.dropna(axis=0,inplace=True)
#axis=0 删除所有缺失的行 axis=1删除所有缺失的列
#参数inplace 为True表示在原数据集上修改，false 复制一个对象修改

y=data2.iloc[:,-1]#要输入的是标签，不是特征矩阵，所以允许一位
le=LabelEncoder()#实例化
le=le.fit(y)
label=le.transform(y)
#print(label.classes_)
le.fit_transform(y)#一步到位
le.inverse_transform(label)#逆转
data=data2.copy()
