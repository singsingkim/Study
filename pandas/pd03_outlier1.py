import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
                      #   25%      50%       75%
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])  # 퍼센트 지점
    print('1사분위 : ', quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1   # 이상치 찾는 인스턴스 정의
    # 최대값이 이상치라면 최대값최소값으로 구하는 이상치는 이상치를 구한다고 할수없다
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 - (iqr * 1.5)
    # -10 의 1.5 범위만큼 과 50의 1.5 범위만큼을 이상치로 생각을 하고 배제
    # 4~10 까지는 안정빵이라고 정의
    
    # 조건문(인덱스 반환) 
    return np.where((data_out>upper_bound) |    # 19보다 크거나
                    (data_out>lower_bound))
    
outliers_loc = outliers(aaa)
print('이상치의 위치 : ', outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()