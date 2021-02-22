from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data=[[-1,2],[-0.5,6],[0,10],[1,18]]

#实现归一化
scaler=MinMaxScaler()
scaler=scaler.fit(data)#生成min和max
result=scaler.transform(data)#导出结果
print(result)
result_=scaler.fit_transform(data)#训练和导出结果(效果和上面两步的一样)
print(result_)
s=scaler.inverse_transform(result)#将归一化之后的结果逆转
print(s)
#使用MinMaxScaler的参数feature_range实现将数据归一化到【0,1】以外的范围中
scaler=MinMaxScaler(feature_range=[5,10])
result=scaler.fit_transform(data)
print(result)
#当x中的特征数量非常多的时候，fit可能计算不了，这个时候使用scaler.partial_fit(data)来计算