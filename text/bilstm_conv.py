"""
This is an LSTM version of this kernel: https://www.kaggle.com/eashish/bidirectional-gru-with-convolution
I merely cleaned it up a bit and replaced the GRU with an LSTM.
"""

import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
###########one hour per epoch


EMBEDDING_FILE = 'glove.840B.300d.txt'
train_x = pd.read_csv('train.csv')
test_x = pd.read_csv('test_1.csv')



max_features=80000
maxlen=150
embed_size=300

train_x['title'].fillna(' ')
test_x['title'].fillna(' ')
train_y = train_x['Category'].values
train_y = to_categorical(train_y)
train_x = train_x['title'].str.lower()

test_y = test_x['Category'].values
test_y = to_categorical(test_y)
test_x = test_x['title'].str.lower()


# Vectorize text + Prepare GloVe Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Build Model
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.35)(x)

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])

out = Dense(58, activation='sigmoid')(x)

model = Model(inp, out)
model = multi_gpu_model(model)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Prediction
batch_size = 128
epochs = 1

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)



score = model.evaluate(test_x, test_y,
                       batch_size=128, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_df = pd.read_csv("test.csv")

test_x = test_df['title']
test_x = tokenizer.texts_to_sequences(test_x)
X_te = sequence.pad_sequences(test_x,maxlen = maxlen)
all_preds = model.predict(X_te)
y_te = [np.argmax(pred) for pred in all_preds]
submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category": y_te})
submit_df.to_csv("submission_bilstmconv.csv", index=False)
