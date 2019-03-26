# Some imports, we are not gong to use all the imports in this workbook but in subsequent workbooks we surely will.
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import multi_gpu_model
from keras.utils import to_categorical


from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf
import os
import time
import gc
import re
import glob


# Define some Global Variables
max_features = 30000 # Maximum Number of words we want to include in our dictionary
maxlen = 72 # No of words in question we want to create a sequence with
embed_size = 300# Size of word to vec embedding we are using
batch_size = 128

# Some preprocesssing that will be common to all the text classification methods you will see. 
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
# def clean_text(x):
#     x = str(x)
#     for punct in puncts:
#         x = x.replace(punct, f' {punct} ')
#     return x

# Loading the data
# test_df = pd.read_csv("test.csv")


# def load_and_prec():

test_df = pd.read_csv("test.csv")

test_x = test_df['title']

train_df = pd.read_csv("traindrop.csv")
# test_df = pd.read_csv(".test.csv")

print("Train shape : ",train_df.shape)
# print("Test shape : ",test_df.shape)

train_df["title"] = train_df["title"]
# test_df["title"] = test_df["title"]

## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=2018) # .08 since the datasize is large enough.

## fill up the missing values
train_X = train_df["title"].fillna("_##_").values
val_X = val_df["title"].fillna("_##_").values
# test_X = test_df["title"].fillna("_##_").values

## Tokenize the sentences
'''
keras.preprocessing.text.Tokenizer tokenizes(splits) the texts into tokens(words).
Signature:
Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs)

The num_words parameter keeps a prespecified number of words in the text only. 
It also filters some non wanted tokens by default and converts the text into lowercase.

It keeps an index of words(dictionary of words which we can use to assign a unique number to a word) 
which can be accessed by tokenizer.word_index.
For example - For a text corpus the tokenizer word index might look like. 
The words in the indexed dictionary are sort of ranked in order of frequencies,
{'the': 1,'what': 2,'is': 3, 'a': 4, 'to': 5, 'in': 6, 'of': 7, 'i': 8, 'how': 9}

The texts_to_sequence function converts every word(token) to its respective index in the word_index

So Lets say we started with 
train_X as something like ['This is a sentence','This is another bigger sentence']
and after fitting our tokenizer we get the word_index as {'this':1,'is':2,'sentence':3,'a':4,'another':5,'bigger':6}
The texts_to_sequence function will tokenize the sentences and replace words with individual tokens to give us 
train_X = [[1,2,4,3],[1,2,5,6,3]]
'''
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
# test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences. We need to pad the sequence with 0's to achieve consistent length across examples.
'''
We had train_X = [[1,2,4,3],[1,2,5,6,3]]
lets say maxlen=6
    We will then get 
    train_X = [[1,2,4,3,0,0],[1,2,5,6,3,0]]
'''

test_x = tokenizer.texts_to_sequences(test_x)
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_x,maxlen = maxlen)
# test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['Category'].values
train_y = to_categorical(train_y)
print('train_y',train_y.shape)
val_y = val_df['Category'].values  
val_y = to_categorical(val_y)
test_y = test_df['Category'].values
test_y = to_categorical(test_y)
#shuffling the data
np.random.seed(2018)
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))

train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]    
    
#     return train_X, val_X, train_y, val_y,test_X,test_y,tokenizer.word_index


# train_X, val_X,  train_y, val_y, test_X,test_y,word_index = load_and_prec()

# Word 2 vec Embedding

def load_glove(word_index):
    '''We want to create an embedding matrix in which we keep only the word2vec for words which are in our word_index
    '''
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

embedding_matrix = load_glove(tokenizer.word_index)

filter_sizes = [1,2,3,5]
num_filters = 100
# https://www.kaggle.com/yekenot/2dcnn-textclassifier
def model_cnn(embedding_matrix):

    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(58, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model = multi_gpu_model(model, gpus=4)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = model_cnn(embedding_matrix)
model.summary()
def train_pred(model, epochs=2):
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    for e in range(epochs):
        model.fit(train_X, train_y, verbose=1,batch_size=128, epochs=1, validation_data=(val_X, val_y),callbacks=callbacks)
    model.load_weights(filepath)
    pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
    # pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y


pred_val_y = train_pred(model, epochs=15)
'''
A function specific to this competition since the organizers don't want probabilities 
and only want 0/1 classification maximizing the F1 score. This function computes the best F1 score by looking at val set predictions
'''

# def f1_smart(y_true, y_pred):
#     thresholds = []
#     for thresh in np.arange(0.1, 0.501, 0.01):
#         thresh = np.round(thresh, 2)
#         res = metrics.f1_score(y_true, (y_pred > thresh).astype(int))
#         thresholds.append([thresh, res])
#         print("F1 score at threshold {0} is {1}".format(thresh, res))

#     thresholds.sort(key=lambda x: x[1], reverse=True)
#     best_thresh = thresholds[0][0]
#     best_f1 = thresholds[0][1]
#     print("Best threshold: ", best_thresh)
#     return  best_f1, best_thresh

# f1, threshold = f1_smart(val_y, pred_val_y)
# print('Optimal F1: {} at threshold: {}'.format(f1, threshold))


# Evaluate the accuracy of our trained model



test_df = pd.read_csv("test.csv")

test_x = test_df['title']
test_x = tokenizer.texts_to_sequences(test_x)
X_te = pad_sequences(test_x,maxlen = maxlen)
all_preds = model.predict(X_te)
y_te = [np.argmax(pred) for pred in all_preds]
submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category": y_te})
submit_df.to_csv("submission_drop.csv", index=False)
