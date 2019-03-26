
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model

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

from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
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

EMBEDDING_FILE = '/home/students/student3_2a/nsdc/text/crawl-300d-2M.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

del all_embs, X_train, X_test, train, test
gc.collect()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
print('preprocessing done')

session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


print("Done loading rnn")
class conv1ddpcnn():
    def __init__(self):
        #model
        #wrote out all the blocks instead of looping for simplicity
        filter_nr = 64
        filter_size = 3
        max_pool_size = 3
        max_pool_strides = 2
        dense_nr = 256
        spatial_dropout = 0.2
        dense_dropout = 0.5
        train_embed = False
        conv_kern_reg = regularizers.l2(0.00001)
        conv_bias_reg = regularizers.l2(0.00001)

        comment = Input(shape=(maxlen,))
        emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
        emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        resize_emb = PReLU()(resize_emb)
            
        block1_output = add([block1, resize_emb])
        block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)
        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)
            
        block2_output = add([block2, block1_output])
        block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)
        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)
            
        block3_output = add([block3, block2_output])
        block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)
        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)

        block4_output = add([block4, block3_output])
        block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

        block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
        block5 = BatchNormalization()(block5)
        block5 = PReLU()(block5)
        block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
        block5 = BatchNormalization()(block5)
        block5 = PReLU()(block5)

        block5_output = add([block5, block4_output])
        block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

        block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
        block6 = BatchNormalization()(block6)
        block6 = PReLU()(block6)
        block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
        block6 = BatchNormalization()(block6)
        block6 = PReLU()(block6)

        block6_output = add([block6, block5_output])
        block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

        block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
        block7 = BatchNormalization()(block7)
        block7 = PReLU()(block7)
        block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
        block7 = BatchNormalization()(block7)
        block7 = PReLU()(block7)

        block7_output = add([block7, block6_output])
        output = GlobalMaxPooling1D()(block7_output)

        output = Dense(dense_nr, activation='linear')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)
        output = Dropout(dense_dropout)(output)
        output = Dense(58, activation='sigmoid')(output)

        model = Model(comment, output)

        model = multi_gpu_model(model)
        model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizers.Adam(),
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
