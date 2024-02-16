import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# 1 데이터
x_datasets = np.array([range(100), range(301, 401)]).T                        # 삼성 종가, 하이닉스 종가

# 행의크기, 데이터의 갯수는 맞춰주어야한다

y1 = np.array(range(3001, 3101))     # 비트코인 종가
y2 = np.array(range(13001, 13101))   # 이더리움 종가

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets,  y1, y2, train_size=0.7, random_state=123)
# 어떤 데이터이든 각각 7:3 으로 짜른다

print(x_train.shape, y1_train.shape, y2_train.shape)        # (70, 2) (70,) (70,)


# 2-1 모델
input1=Input(shape=(2,))
dense1=Dense(10, activation='relu', name='bit1')(input1)
dense2=Dense(10, activation='relu', name='bit2')(dense1)
dense3=Dense(10, activation='relu', name='bit3')(dense2)
output=Dense(10, activation='relu', name='bit4')(dense3)


# 2-3-y1 concatnate
# merge1 = concatenate(output, name='mg1')
merge1 = output
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)

# 2-3-y2 concatnate
# merge11 = concatenate(output, name='mg11')
merge11 = output
merge12 = Dense(7, name='mg12')(merge11)
merge13 = Dense(11, name='mg13')(merge12)
last_output2 = Dense(1, name='last2')(merge13)


model = Model(inputs=input1, outputs=[last_output1, last_output2])     # 두 개 이상은 리스트

# model.summary()

# 3 컴파일, 훈련
# 맹글
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
            mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
            patience=100,      # 최소값 찾은 후 열 번 훈련 진행
            verbose=1,
            restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
            )

model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train],
          epochs=2000,
          batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

'''
'''

# 4 예측, 평가
loss = model.evaluate(x_test, [y1_test, y2_test])
y_predict = model.predict(x_test)
# print("="*100)
# print(y_predict[0])
# print("="*100)
r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print('로스 : ', loss)
print('r2_1 : ', r2_1)
print('r2_2 : ', r2_2)

# 로스 :  [1.177002191543579, 0.8220102787017822, 0.3549918532371521]
# r2_1 :  0.9989404292781001
# r2_2 :  0.9995424157218724