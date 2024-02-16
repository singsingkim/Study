import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# 1 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T                        # 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T  # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).T                        # 삼성 종가, 하이닉스 종가

# 행의크기, 데이터의 갯수는 맞춰주어야한다

print(x1_datasets.shape, x2_datasets.shape, x3_datasets.shape)
# (100, 2) (100, 3) (100, 4)
y = np.array(range(3001, 3101))     # 비트코인 종가

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, train_size=0.7, random_state=123)
# 어떤 데이터이든 각각 7:3 으로 짜른다

print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape)
# (70, 2) (70, 3) (70, 4) (70,)



# 2-1 모델
input1=Input(shape=(2,))
dense1=Dense(10, activation='relu', name='bit1')(input1)
dense2=Dense(10, activation='relu', name='bit2')(dense1)
dense3=Dense(10, activation='relu', name='bit3')(dense2)
output1=Dense(10, activation='relu', name='bit4')(dense3)

# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()


# 2-2 모델
input11=Input(shape=(3,))
dense11=Dense(100, activation='relu', name='bit11')(input11)
dense12=Dense(100, activation='relu', name='bit12')(dense11)
dense13=Dense(100, activation='relu', name='bit13')(dense12)
output11=Dense(5, activation='relu', name='bit14')(dense13)

# model12 = Model(inputs=input11, outputs=output11)
# model12.summary()

# 2-3 모델
input21=Input(shape=(4,))
dense21=Dense(100, activation='relu', name='bit21')(input21)
dense22=Dense(100, activation='relu', name='bit22')(dense21)
dense23=Dense(100, activation='relu', name='bit23')(dense22)
output21=Dense(5, activation='relu', name='bit24')(dense23)

# model12 = Model(inputs=input11, outputs=output11)
# model12.summary()


# 2-4 concatnate
merge1 = concatenate([output1, output11, output21], name='mg1')
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)


model = Model(inputs=[input1, input11, input21], outputs=last_output)     # 두 개 이상은 리스트

model.summary()

# 3 컴파일, 훈련
# 맹글
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
            mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
            patience=100,      # 최소값 찾은 후 열 번 훈련 진행
            verbose=1,
            restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
            )

model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train,x2_train,x3_train], y_train,
          epochs=1000,
          batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

# 4 예측, 평가
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
y_predict = model.predict([x1_test, x2_test, x3_test])
r2 = r2_score(y_test, y_predict)
print('로스 : ', loss)
print('R2 : ', r2)

# 로스 :  0.04485968127846718
# R2 :  0.9999421758977535
