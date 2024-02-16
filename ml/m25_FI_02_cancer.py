import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Data
# x,y = load_breast_cancer(return_X_y=True)

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# =======================================1=====
''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.03546066 0.03446177 0.43028001 0.49979756"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)
# ========================================================


print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 2. Model

rs = 1212
models =[DecisionTreeClassifier(random_state=rs),
    RandomForestClassifier(random_state=rs),
    GradientBoostingClassifier(random_state=rs),
    XGBClassifier(random_state=rs)]

for model in models:
    model.fit(x_train,y_train)
    result= model.score(x_test,y_test)
    print("model.score:",result)
    y_predict=model.predict(x_test)
    acc=accuracy_score(y_test,y_predict)
    # print(model, "acc", acc)
    # print(model.feature_importances_) 
    print(type(model).__name__, ":",model.feature_importances_)   
        




# print(model, "acc", acc)

# print(model.feature_importances_)   
#[0.04519231 0.         0.54879265 0.40601504] - 각각 feature 점수
# print(model, ":",model.feature_importances_)   
#DecisionTreeClassifier(random_state=1212) : [0.04519231 0.         0.54879265 0.40601504]
#RandomForestClassifier(random_state=1212) : [0.09680396 0.02696853 0.39722367 0.47900384]
#GradientBoostingClassifier(random_state=1212) : [0.01522103 0.0134919  0.27358378 0.6977033 ]
# XGBClassifier(random_state=1212) : [0.01192079 0.02112738 0.51003134 0.45692044]