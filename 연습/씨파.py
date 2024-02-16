import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

str = "저는 인공지능에 관심이 많습니다."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([str])
print(tokenizer.word_index)

print(np.unique(str))
print(pd.value_counts(str))