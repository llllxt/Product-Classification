from init import bilstm
from fasttext import fasttexts
from rnn import deepModel, Finisher
from gruwithft import gruwithft
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

EMBEDDING_FILE = '/home/students/student3_2a/nsdc/text/glove.840B.300d.txt'
train_x = pd.read_csv('/home/students/student3_2a/nsdc/text/train.csv')
test_x = pd.read_csv('/home/students/student3_2a/nsdc/text/test.csv')



max_features=80000
maxlen=150
embed_size=300

x_train = train_x['title'].fillna(' ')
train_y = train_x['Category'].values
y_train = to_categorical(train_y)


test_x['title'].fillna(' ')

x_test = test_x['title'].str.lower()


# # Vectorize text + Prepare GloVe Embedding
# tokenizer = text.Tokenizer(num_words=max_features, lower=True)
# tokenizer.fit_on_texts(list(train_x))

# train_x = tokenizer.texts_to_sequences(train_x)
# test_x = tokenizer.texts_to_sequences(test_x)

# x_train = sequence.pad_sequences(train_x, maxlen=maxlen)
# x_test = sequence.pad_sequences(test_x, maxlen=maxlen)

class Ensemble():
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):
        
        folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=2017).split(X))
        S_train = np.zeros((X.shape[0], 58*len(self.base_models)))
        S_test = []
        final_test = np.empty((T.shape[0],58*len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(folds):
                print(train_idx)
                print("On {}{}".format(i,j))
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_pred, y_test = clf.run(X_train,y_train,X_holdout,T)[:]
                S_train[test_idx, i*58:i*58+58] = y_pred
                S_test.append(y_test)
            final_test[:,i*58:i*58+58] = np.mean(np.array(S_test), axis=0)
        return self.stacker.finish(np.expand_dims(S_train, axis=2), y, np.expand_dims(final_test, axis=2))[:]

go = Ensemble(5, Finisher((174,1)), [bilstm(),deepModel(),gruwithft()])
go.fit_predict(x_train, y_train, x_test)
