from sklearn.preprocessing import StandardScaler
data=[[-1,2],[-0.5,6],[0,10],[1,18]]
scaler=StandardScaler()
scaler.fit(data)#本质是生成均值和方差
print("均值是：{}".format(scaler.mean_))
print("方差是：{}".format(scaler.var_))
x_std=scaler.transform(data)#导出结果
print(x_std)
print("均值是：{}".format(x_std.mean()))
print("方差是：{}".format(x_std.std()))
scaler.fit_transform(data)#一步达成结果
scaler.inverse_transform(x_std)#逆转标准化
