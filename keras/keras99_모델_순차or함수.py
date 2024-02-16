# 모델 순차형
model = Sequential()
model.add(Dense(100, input_shape=(784,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 모델 함수형
input1 = Input(shape=(28*28,))
dense1 = Dense(100, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(10, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(100, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(100, activation='relu')(drop3)
dense5 = Dense(60, activation= 'relu')(dense4)
drop4 = Dropout(0.3)(dense5)
dense6 = Dense(30, activation= 'relu')(drop4)
output1 = Dense(10, activation= 'softmax')(dense6)
model = Model(inputs=input1, outputs=output1)


# 컨벌레이션
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape = (300, 300, 3),
                 strides=2, padding='same')) 
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), activation='relu')) #전달 (N,25,25,10)
model.add(Conv2D(128,(2,2))) #전달 (N,22,22,15)
model.add(MaxPooling2D())
model.add(Flatten()) #평탄화
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
