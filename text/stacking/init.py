
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
from keras.utils import multi_gpu_model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, BatchNormalization, GRU, Bidirectional, LSTM, GlobalMaxPool1D,Flatten
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

EMBEDDING_FILE = '/home/students/student3_2a/nsdc/text/glove.840B.300d.txt'
train_x = pd.read_csv('/home/students/student3_2a/nsdc/text/train.csv')
test_x = pd.read_csv('/home/students/student3_2a/nsdc/text/test.csv')



max_features=80000
maxlen=72
embed_size=300


train_y = train_x['Category'].values
train_y = to_categorical(train_y)
train_x = train_x['title'].str.lower()

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


print("Done loading rnn")
class bilstm():
    def __init__(self):
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

        self.model = Model(inp, out)
        self.model = multi_gpu_model(self.model)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        
    def run(self, X,y,X_te,full_X_te):
        print("Running LSTM...")
        # X = np.array(X)
        # y = np.array(y)
        # X_te = np.array(X_te)
        x_train = pad_sequences(tokenizer.texts_to_sequences(X),maxlen=maxlen)
        x_test = pad_sequences(tokenizer.texts_to_sequences(X_te),maxlen=maxlen)
        full_x_test = pad_sequences(tokenizer.texts_to_sequences(full_X_te), maxlen=maxlen)
        y = y
        print("Formatted inputs. Fitting...")
        file_path="weights_base.best.hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
        callbacks_list = [checkpoint, early]
        #add validation split to test
        self.model.fit(x_train,y, batch_size=128, epochs = 2, callbacks = callbacks_list,validation_split = 0.2,verbose=1)
        self.model.load_weights(file_path)
        y_pred = self.model.predict([x_test],batch_size=1024, verbose=1)
        y_test = self.model.predict([full_x_test], batch_size=1024, verbose=1)
        # sample_submission = pd.read_csv("sample_submission.csv").head(len(y_test))
        # sample_submission[classes] = y_test
        print("LSTM done")
        return y_pred,y_test
