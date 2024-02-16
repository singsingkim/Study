# 임배딩 레이어는 데이터를 벡터화 시켜서 돌린후 3차원으로 배출한다

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# 1 데이터
docs= [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요', 
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요', '상헌이 천재',
    '반장 잘생겼다', '욱이 또 잔다',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6,
# '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, 
# '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17,
# '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, 
# '재미없다': 22, '재밌네요': 23, '상헌이': 24, '천재': 25, '반장': 26,
# '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30} # 단어 사전의 갯수 30개

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

print(type(x))  # <class 'list'>

# x = np.array(x) # 최대 리스트 갯수가 5개 이기때문에 2개 짜리도 5 로 맞춰야 가능함
                  # 안되는거 보여주려고 쓴거
                
                
from keras.utils import pad_sequences

pad_x = pad_sequences(x, 
                      padding='pre',    # 0 으로 채우기 : 디폴트 pre
                      maxlen=5,         # 숫자 만큼 열을 만든다
                      truncating='post' # 넘치는 데이터에서 자른다 : 디폴트 pre
                      )
print(pad_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
print(pad_x.shape)  # (15, 5)  
# pad_x = pad_x.reshape(-1,5,1)
print(pad_x.shape)  # (15, 5, 1)  

word_size = len(token.word_index) + 1 
print(word_size)

# 2 모델
###################### 임배딩1
model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
# 임배딩연산량 = input_dim * ouput_dim = 31 * 100 = 3100
# 임배딩의 인풋의 shape : 2차원 , 임배딩의 아웃풋의 shape : 3차원
# 레이어 연산식 = 4 * 10 * (10+100+1)
# 단어사전의 개수 + 1, 
# 출렷노드의갯수, 아웃으로 보낼 Dense(조절 가능)
# 단어의 길이 (패딩으로 늘림 -> 5개)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 7,625
# Trainable params: 7,625
# Non-trainable params: 0


###################### 임배딩2
model.add(Embedding(input_dim=31, output_dim=100, ))
model.add(LSTM(10, input_shape=(5, 1)))
model.add(Dense(7,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 7,625
# Trainable params: 7,625
# Non-trainable params: 0

###################### 임배딩3
model.add(Embedding(input_dim=20, output_dim=100, ))
# input_dim = 31    # 디폴트
# input_dim = 20    # 단어사전의 갯수보다 작을때    # 임의로 단어사전 삭제(연산량 감소)  # 성능 저하
# input_dim = 41    # 단어사전의 갯수보다 많을때    # 임의로 단어사전 증폭(연산량 증가)  # 성능 저하
model.add(LSTM(10, input_shape=(5, 1)))
model.add(Dense(7,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


###################### 임배딩4
# model.add(Embedding(31, 100, ))               # 잘 돌아감
# model.add(Embedding(31, 100, 5))              # 에러
model.add(Embedding(31, 100, input_length=5))   # 잘 돌아감
model.add(LSTM(10, ))   # input_shape 생략 가능
model.add(Dense(7,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()



'''

# 3 컴파일, 훈련
es = EarlyStopping(
    monitor='loss', mode='min', patience=1000, 
    verbose=1, restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=5000, 
          batch_size=32, verbose=1, callbacks=[es])

# 4 평가, 예측
results = model.evaluate(pad_x, labels)
y_predict = model.predict(pad_x)

print('로스 : ', results[0])
print('ACC : ', results[1])

# 로스 :  3.700529020989052e-07
# ACC :  1.0

'''
