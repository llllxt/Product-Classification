'''
#This example demonstrates the use of fasttext for text classification
Based on Joulin et al's paper:
[Bags of Tricks for Efficient Text Classification
](https://arxiv.org/abs/1607.01759)
Results on IMDB datasets with uni and bi-gram embeddings:
Embedding|Accuracy, 5 epochs|Speed (s/epoch)|Hardware
:--------|-----------------:|----:|:-------
Uni-gram |            0.8813|    8|i7 CPU
Bi-gram  |            0.9056|    2|GTx 980M GPU
'''
import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from tqdm import tqdm


import numpy as np
import os
import pandas as pd

from collections import defaultdict

import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model




# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 80000
maxlen = 72
batch_size = 128
embedding_dims = 50
epochs = 50
vocab_size = 30000


print('Loading data...')
df = pd.read_csv("fashion.csv")
test = pd.read_csv("fashion_ans.csv")
x_test = test['title'].values
y_test = test['Category'].values


x = df['title']
y = df['Category'].values

y = to_categorical(y)

def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams

    docs = []
    for doc in df.title:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))

    return docs

min_count = 2

docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

X_test = tokenizer.texts_to_sequences(x_test)
Y_test = to_categorical(y_test)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
maxlen = 256

docs = pad_sequences(sequences=docs, maxlen=maxlen)


x_train, x_val, y_train, y_val = train_test_split(docs, y, test_size=0.2, random_state=0)



checkpoint = ModelCheckpoint('fasttext.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

input_dim = np.max(docs) + 1
embedding_dims = 20

def create_model(embedding_dims=20, optimizer='adam'):
    # if(os.path.exists('fasttext.h5')):
    #     print('load model from fasttext')
    #     model = load_model('fasttext.h5')
    # else:
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(14, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


model = create_model()
hist = model.fit(x_train, y_train,
                batch_size=128,
                validation_data=(x_val, y_val),
                epochs=epochs,
                class_weight = 'auto',
                callbacks = [checkpoint,early])


# print('Build model...')
# model = Sequential()
#
# # we start off with an efficient embedding layer which maps
# # our vocab indices into embedding_dims dimensions
# model.add(Embedding(max_features,
#                     embedding_dims,
#                     input_length=maxlen))
#
# # we add a GlobalAveragePooling1D, which will average the embeddings
# # of all words in the document
# model.add(GlobalAveragePooling1D())
#
# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(58, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose = 1,
#           validation_data=(x_test, y_test),
#           class_weight = 'auto',
#           callbacks=[checkpoint,early])

# Evaluate the accuracy of our trained model
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


y_softmax = model.predict(X_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(Y_test)):
    probs = Y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)
    
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:100]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (100 - len(embeds))
    return np.array(embeds)
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr


test_df = pd.read_csv("test_fashion.csv")

test_x = test_df['title']
test_x = tokenizer.texts_to_sequences(test_x)
X_te = pad_sequences(test_x,maxlen = maxlen)
all_preds = model.predict(X_te)
y_te = [np.argmax(pred) for pred in all_preds]
submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category": y_te})
submit_df.to_csv("submission(fasttext).csv", index=False)

#
#cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
#plt.figure(figsize=(24,20))
#plot_confusion_matrix(cnf_matrix, classes=range(0,59), title="Confusion matrix")
#plt.show()
#
