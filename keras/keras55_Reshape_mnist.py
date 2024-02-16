import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,Reshape,Conv1D,LSTM
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint

#1.data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(x_train)
# print(x_train[0])
print(y_train[0])   #5
print(np.unique(y_train,return_counts=True))    #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]


import datetime
date=datetime.datetime.now()
print(date) #2024-01-17 10:54:36.094603 - 
#월,일,시간,분 정도만 추출
print(type(date))   #<class 'datetime.datetime'>
date=date.strftime("%m%d-%H%M")
#%m 하면 month를 땡겨옴, %d 하면 day를 / 시간,분은 대문자
print(date) #0117_1058
print(type(date))   #<class 'str'> 문자열로 변경됨

path='../_data/_save/MCP/'  #문자를 저장
filename= '{epoch:04d}-{val_loss:.4f}.hdf5' #0~9999 : 4자리 숫자까지 에포 / 0.9999 소숫점 4자리 숫자까지 발로스
filepath= "".join([path,'k31',date,'_',filename]) # ""는 공간을 만든거고 그안에 join으로 합침 , ' _ ' 중간 공간


#3차원 4차원으로 변경
x_train=x_train.reshape(60000,28,28,1)  #data 내용,순서 안바뀌면 reshape 가능

# x_test=x_test.reshape(10000,28,28,1)  #아래와 같다 - 값을 모를때 적용
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)
ohe=OneHotEncoder(sparse=False)
# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)
y_train=ohe.fit_transform(y_train)
y_test=ohe.transform(y_test)


# x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.8,random_state=112,
                                            #    stratify=y_train)



#2.model
model = Sequential()                            # Output Shape
model.add(Dense(9,input_shape=(28,28,1)))       # (None, 28, 28, 9)
model.add(Conv2D(10, (3,3)))                    # (None, 26, 26, 10) 
model.add(Reshape(target_shape=(26*26,10)))     # (None, 676, 10) 
model.add(Conv1D(15,4))                         # (None, 673, 15)
model.add(LSTM(8,return_sequences=True))        # (None, 673, 8)
model.add(Conv1D(14,2))                         # (None, 672, 14) 
model.add(Dense(8))                             # (None, 672, 8)   
model.add(Dense(7,input_shape=(8,)))            # (None, 672, 7) 
model.add(Flatten())                            # (None, 4704)
model.add(Dense(6))                             # (None, 6) 
model.add(Dense(10,activation='softmax'))       # (None, 10) 

model.summary()

# Dense(9, input_shape=(28,28,1)): 이 레이어는 28x28 크기의 이미지를 입력으로 받고, 9개의 뉴런을 가진 완전히 연결된 층입니다. 따라서 출력 형태는 (None, 28, 28, 9)가 됩니다. 여기서 None은 배치 크기를 의미합니다.

# Conv2D(10, (3,3)): 3x3 크기의 필터를 사용하여 10개의 컨볼루션 필터를 적용합니다. 이는 입력 이미지의 차원을 변경시킵니다. 출력 형태는 (None, 26, 26, 10)입니다. 이는 컨볼루션 레이어를 통과한 후 이미지의 크기가 3씩 줄어들었고, 10개의 필터로 인해 깊이가 10이 되었음을 의미합니다.

# Reshape(target_shape=(26*26,10)): 3차원 데이터를 2차원으로 변환합니다. 이 레이어는 이전 레이어의 출력인 (None, 26, 26, 10)을 (None, 676, 10)으로 변환합니다.

# Conv1D(15,4): 이 레이어는 1차원 컨볼루션을 수행합니다. 입력의 길이를 변경시키고, 여기서는 4개의 윈도우 크기를 사용하여 15개의 필터를 적용합니다. 출력 형태는 (None, 673, 15)가 됩니다.

# LSTM(8, return_sequences=True): LSTM 셀을 사용하여 순차 데이터를 처리합니다. 이 레이어는 이전 레이어의 출력인 (None, 673, 15)을 입력으로 받아 순차 데이터를 처리하고, 출력 형태는 (None, 673, 8)이 됩니다.

# Conv1D(14,2): 다시 1차원 컨볼루션을 수행합니다. 입력의 길이를 변경시키고, 2개의 윈도우 크기를 사용하여 14개의 필터를 적용합니다. 출력 형태는 (None, 672, 14)가 됩니다.

# Dense(8): 8개의 뉴런을 가진 완전히 연결된 층입니다. 입력 형태는 (None, 672, 14)이며, 출력 형태는 (None, 672, 8)이 됩니다.

# Dense(7, input_shape=(8,)): 7개의 뉴런을 가진 완전히 연결된 층입니다. 입력 형태가 명시적으로 주어지며, 이전 레이어의 출력 (None, 672, 8)에서 (None, 8)로 변경됩니다. 출력 형태는 (None, 672, 7)이 됩니다.

# Flatten(): 이전 레이어의 출력을 1차원으로 평평하게 만듭니다. 따라서 (None, 672, 7)에서 (None, 4704)로 변경됩니다.

# Dense(6): 6개의 뉴런을 가진 완전히 연결된 층입니다. 입력 형태는 (None, 4704)이며, 출력 형태도 (None, 6)이 됩니다.

# Dense(10, activation='softmax'): 마지막으로, 10개의 클래스에 대한 확률을 출력하기 위해 소프트맥스 활성화 함수가 있는 완전히 연결된 층을 추가합니다. 출력 형태는 (None, 10)이 됩니다.

'''
#3.compile,fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=120,restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,
                    filepath=filepath   #경로저장
                    )
model.fit(x_train,y_train,epochs=15,batch_size=88,verbose=1,validation_split=0.2,
          callbacks=[es,mcp])

#4.evaluate,predict
results=model.evaluate(x_test,y_test)
print("loss:",results[0])
print("acc:",results[1])

# loss: 0.1465 - acc: 0.9843    
# loss: 0.1465291827917099     
# acc: 0.9843000173568726  
'''