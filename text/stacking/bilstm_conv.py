
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
#Get embed matrix from embedfile
def load_glove(word_index):
    '''We want to create an embedding matrix in which we keep only the word2vec for words which are in our word_index
    '''
    EMBEDDING_FILE = '/home/students/student3_2a/nsdc/text/glove.840B.300d.txt'
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

print("Done loading rnn")
class deepModel():
    def __init__(self):
        filter_sizes = [1,2,3,5]
        num_filters = 100
        inp = Input(shape=(maxlen,))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
        x = Reshape((maxlen, embed_size, 1))(x)

        maxpool_pool = []
        for i in range(len(filter_sizes)):
            conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                         kernel_initializer='he_normal', activation='relu')(x)
            maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

        z = Concatenate(axis=1)(maxpool_pool)   
        z = Flatten()(z)

        z = Dropout(0.1)(z)

        outp = Dense(58, activation="sigmoid")(z)

        self.model = Model(inputs=inp, outputs=outp)
        self.model = multi_gpu_model(self.model, gpus=4)
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
class Finisher():
    def __init__(self,shape):
        self.model = Sequential()
        self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu',input_shape = shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(58, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    def finish(self,X,y,t):
        test_df = pd.read_csv("/home/students/student3_2a/nsdc/text/test.csv")

        file_path="weights_base.second.hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        callbacks_list = [checkpoint, early]
        #add validation split to test
        self.model.fit(X,y, batch_size=128, epochs = 20, callbacks = callbacks_list,validation_split = 0.2, verbose=1)
        self.model.load_weights(file_path)
        all_preds = self.model.predict([t],batch_size=1024, verbose=1)
        y_te = [np.argmax(pred) for pred in all_preds]
        submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category": y_te})
        submit_df.to_csv("submission_ensemble.csv", index=False)
        return
        # c = pd.read_csv("/data/sample_submission.csv")
        # c[classes] = y_pred
        # return c
