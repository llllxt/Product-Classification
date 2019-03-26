
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, BatchNormalization, GRU, Bidirectional, LSTM, GlobalMaxPool1D,Flatten
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from keras.callbacks import *
from sklearn.utils import class_weight
#io

max_features = 76670 # Maximum Number of words we want to include in our dictionary
maxlen = 72 # No of words in question we want to create a sequence with
embed_size = 300# Size of word to vec embedding we are using
batch_size = 128

train_df = pd.read_csv('/home/students/student3_2a/nsdc/text/train.csv')
x_train = train_df['title'].fillna("_na_").values

output_file = "submission.csv"
classes = train_df['Category'].values
classes = to_categorical(classes)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(x_train))

EMBEDDING_FILE = '/home/students/student3_2a/nsdc/text/crawl-300d-2M.vec'

max_features = 70000
maxlen = 72
embed_size = 300

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

print("Done loading rnn")
class gruwithft():
    def __init__(self):
        inp = Input(shape=(maxlen, ))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(GRU(80, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(58, activation="sigmoid")(conc)
        
        self.model = Model(inputs=inp, outputs=outp)
        self.model = multi_gpu_model(self.model, gpus=4)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
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
        file_path="gruwithft.best.hdf5"
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
