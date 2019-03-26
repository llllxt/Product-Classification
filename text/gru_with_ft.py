import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from keras.callbacks import *
from sklearn.utils import class_weight


EMBEDDING_FILE = 'crawl-300d-2M.vec'

train = pd.read_csv('train.csv')
test = pd.read_csv('test_1.csv')

X_train = train["title"].fillna("fillna").values
y_train = train["Category"].values
classweight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

y_train = to_categorical(y_train)
X_test = test["title"].fillna("fillna").values
y_test = test['Category'].values
y_test = to_categorical(y_test)

max_features = 70000
maxlen = 72
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


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


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(58, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model = multi_gpu_model(model, gpus=4)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()


batch_size = 128
epochs = 8

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
filepath="weights_best_gru_with_ft.h5"
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, class_weight = classweight,validation_data=(X_val, y_val),
                 callbacks=[RocAuc,checkpoint,reduce_lr], verbose=1)

score = model.evaluate(x_test, y_test,
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
submit_df.to_csv("submission_gruwithft.csv", index=False)
