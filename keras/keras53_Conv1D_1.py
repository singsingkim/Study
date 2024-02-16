import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM , Conv1D, Flatten # 차원 하나만 작다
from keras.callbacks import EarlyStopping
# 1 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])  
print(x.shape, y.shape) # (7, 3) (7,)
    # 3차원 모양 맞춰주기 위해 리쉐이프 필요

x = x.reshape(7,3,1)
print(x.shape, y.shape) # (7, 3, 1) (7,)

# 2 모델
model = Sequential()
# model.add(SimpleRNN(units=512, input_shape=(3,1))) # (timesteps, features)
# 3-D tensor with shape (batch_size, timesteps, features).
# model.add(LSTM(units=10, input_shape=(3,1))) # (timesteps, features)
model.add(Conv1D(filters=256, kernel_size=2, input_shape=(3, 1)))
model.add(Flatten())    # conv1D 때문에 3차원 데이터로 댄스에 입력되서 2개값이 나온다
model.add(Dense(16))
model.add(Dense(128))
model.add(Dense(6))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# lstm = 565
# con1d = 185

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x, y, epochs=10000, callbacks=[es])

# 4 평가, 예측
results  = model.evaluate(x,y)
print('로스는 : ',results)
y_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(y_pred)
# (3,) -> (1, 3, 1)
print('[8,9,10] 의 결과 : ', y_pred)

# 로스는 :  7.795668949782397e-13
# 1/1 [==============================] - 0s 81ms/step
# [8,9,10] 의 결과 :  [[10.999999]]