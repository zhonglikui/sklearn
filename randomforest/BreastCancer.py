from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=load_breast_cancer()
print(data.data.shape)
rfc=RandomForestRegressor(n_estimators=100,random_state=1)
score_pre=cross_val_score(rfc,data.data,data.target,cv=10).mean()
print(score_pre)

scorel=[]
for i in range(50,70):
    rfc=RandomForestRegressor(n_estimators=i+1,n_jobs=-1,random_state=90)
    score=cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()
