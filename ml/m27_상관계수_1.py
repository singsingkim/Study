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
x, y=load_iris(return_X_y=True)

datasets = load_iris()
x = datasets.data
y = datasets['target']

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)
df['Target(Y)'] = y
print(df)
print('===================================상관계수==============================')
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), 
            annot=True, 
            square=True,    # 표 안에 수치 명시
            cbar=True       # 사이드바 표시
            )
plt.show()


print(matplotlib.__version__)   # 3.8.0 // 3.7.2 에서 수치가 정상적으로 보임