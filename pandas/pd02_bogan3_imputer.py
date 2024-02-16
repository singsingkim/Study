import numpy as np
import pandas as pd
data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
# print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()   # 디폴트 평균
data2 = imputer.fit_transform(data)
print(data2)

imputer = SimpleImputer(strategy='mean')   # 디폴트 평균
data3 = imputer.fit_transform(data)
print(data3)

imputer = SimpleImputer(strategy='median')   # 중위
data4 = imputer.fit_transform(data)
print(data4)

imputer = SimpleImputer(strategy='most_frequent')   # 가장 자주 나오는 놈
data5 = imputer.fit_transform(data)
print(data5)

imputer = SimpleImputer(strategy='constant')   # 상수( 디폴트 0)
data6 = imputer.fit_transform(data)
print(data6)

imputer = SimpleImputer(strategy='constant',
                        fill_value=777)   # 상수( 디폴트 0)
data7 = imputer.fit_transform(data)
print(data7)

##########################################################################
imputer = KNNImputer()  # 가까운곳에 어떤놈이 얼마나 있는지
data8 = imputer.fit_transform(data)
print(data8)

imputer = IterativeImputer()  # 선형회귀알고리즘 = interpolate 와 비슷
data9 = imputer.fit_transform(data)
print(data9)

print(np.__version__)   # 1.26.3 이었던걸 numpy 1.22.4 버전을 재설치
print(np.__version__)   # 1.22.4
# pip install impyute
from impyute.imputation.cs import mice
aaa = mice(data.values, n= 10, seed=777)
print(aaa)

