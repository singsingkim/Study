from sklearn.preprocessing import MinMaxScaler, StandardScaler
############ 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.
############ 스케일링 1-2
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
############ 스케일링 2-1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
############ 스케일링 2-2 (standard)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
# 이미지에서는 데이터값이 0~255 밖에 안되기때문에 초반 과정에서 스케일링1 과정을 해도 MinMaxScaler 적용한것과 같다

