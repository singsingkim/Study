from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# ★ 코드 완성 했을때 오류가 생길텐데 케라스 판다스 사이킷런 전처리과정에서 생길예정
# y의 시작값이 1부터 7까지의 6 개인 이유로 오류가 발생할것
# 오류가 나는 코드는 수정해서 완료시킬것