import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'
# aaa = CustomXGBClassifier()

# 1. Data
# x, y=load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)

# x = np.delete(x, 0, axis=1)
print(x)



### 판다스로 바꿔서 컬럼 삭제 ###
# pd.DataFrame  
# 컬렴명 : datasets.feature_names 안에 있다
### 실습
### 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
### 데이터셋 재구성후
### 기존 모델결과와 비교!!

x = pd.DataFrame(x)
print(x)

# 컬럼 확인
feature_names = datasets.feature_names
print(feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_names_combined = ', '.join(datasets.feature_names)
print(feature_names_combined)
# sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)


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






# 트레인
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)




# 2. Model

RS = 123
models =[DecisionTreeClassifier(random_state=RS),
    RandomForestClassifier(random_state=RS),
    GradientBoostingClassifier(random_state=RS),
    # XGBClassifier(random_state=RS)
    CustomXGBClassifier(random_state=RS)
    ]

for model in models:
    model.fit(x_train,y_train)
    result= model.score(x_test,y_test)
    print("model.score:",result)
    y_predict=model.predict(x_test)
    acc=accuracy_score(y_test,y_predict)
    # print(model, "acc", acc)
    # print(model.feature_importances_) 
    # print(type(model).__name__, ":",model.feature_importances_)   
    print(model, ":", model.feature_importances_)   


