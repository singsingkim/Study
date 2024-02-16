# 5일분(720행) 을 훈련시켜서
# 하루(144행) 뒤를 예측

# 열 14개
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, r2_score
import time
start_time = time.time()

#1. 데이터
np_path = 'c:/_data/_save_npy//'
x_train = np.load(np_path + 'keras52_2_x_train.npy')
y_train = np.load(np_path + 'keras52_2_y_train.npy')
x_test = np.load(np_path + 'keras52_2_x_test.npy')
y_test = np.load(np_path + 'keras52_2_y_test.npy')

print(x_train.shape, y_train.shape) # (335749, 720, 14) (335749,)
print(x_test.shape, y_test.shape)   # (83938, 720, 14) (83938,)

#2. 모델 구성 
model = Sequential()
# model.add(GRU(units=256, return_sequences=True, input_shape=(720, 14))) # (timesteps, features)
model.add(Conv1D(256, 3, input_shape=(720, 14)))    # 댄스에 줄 떄 리턴스퀀스 트루 주게되면 3차원으로 들어가기 때문에 해제해야한다
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,))

model.summary()


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=50,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, epochs=100, 
                batch_size = 4096,
                validation_split=0.2,
                callbacks=[es],
                verbose=1
                )

end_time = time.time()   #끝나는 시간

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("걸린시간 : ", round(end_time - start_time, 2),"초")
print('로스 : ', results[0])

r2 = r2_score(y_test,y_predict)
print('r2 : ', r2)


# 걸린시간 :  10060.68 초
# 로스 :  129.5284423828125
# r2 :  -0.9739202059475607