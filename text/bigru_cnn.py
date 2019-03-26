import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

###########30min per epoch#############
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


train = pd.read_csv("train.csv")
test = pd.read_csv("test_1.csv")
# embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embedding_path = "glove.840B.300d.txt"
embed_size = 300
max_features = 75720
max_len = 150


y = train['Category'].values
y = to_categorical(y)

y_test = test['Category'].values
y_test = to_categorical(y_test)
train["title"].fillna("no comment")
test["title"].fillna("no comment")
X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)


raw_text_train = X_train["title"].str.lower()
raw_text_valid = X_valid["title"].str.lower()
raw_text_test = test["title"].str.lower()

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text_train)
X_train["comment_seq"] = tk.texts_to_sequences(raw_text_train)
X_valid["comment_seq"] = tk.texts_to_sequences(raw_text_valid)
test["comment_seq"] = tk.texts_to_sequences(raw_text_test)

X_train = pad_sequences(X_train.comment_seq, maxlen = max_len)
X_valid = pad_sequences(X_valid.comment_seq, maxlen = max_len)
test = pad_sequences(test.comment_seq, maxlen = max_len)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(units, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dense(58, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model = multi_gpu_model(model)
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 4, validation_data = (X_valid, Y_valid), 
                        verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model
model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

test = pd.read_csv("test_1.csv")
test_x = test_df['title']
test_x = tk.texts_to_sequences(test_x)
test_x =pad_sequences(test_x,maxlen = max_len)

score = model.evaluate(test_x, y_test,
                       batch_size=128, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_df = pd.read_csv("test.csv")

test_x = test_df['title']
test_x = tk.texts_to_sequences(test_x)
X_te = pad_sequences(test_x,maxlen = max_len)
all_preds = model.predict(X_te)
y_te = [np.argmax(pred) for pred in all_preds]
submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category": y_te})
submit_df.to_csv("submission_dpcnn.csv", index=False)
