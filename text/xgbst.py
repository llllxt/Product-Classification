# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
import re, string
from sklearn.metrics import log_loss


# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.fillna("unknown")
test = test.fillna("unknown")


train_mes, valid_mes, train_l, valid_l = train_test_split(train['title'],train['Category'], test_size=0.2, random_state=2)


'''
def text_process(comment):
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
'''
#Couldnt remove the stop words using the above function since it is taking too long
#Can try it on a local machine, I feel it improves the score-Not sure though


'''
transform_com = CountVectorizer().fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
comments_train = transform_com.transform(train['comment_text'])
comments_test = transform_com.transform(test['comment_text'])
gc.collect()'''


def tokenize(s): return re_tok.sub(r' \1 ', s).split()

#Using the tokenize function from Jeremy's kernel
re_tok = re.compile('([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
transform_com = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1).fit(pd.concat([train['title'],test['title']],axis=0))


comments_train = transform_com.transform(train_mes)
comments_valid = transform_com.transform(valid_mes)
comments_test = transform_com.transform(test['title'])
gc.collect()



import xgboost as xgb


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=300):
    param = {}
    param['objective'] = 'multi:softprob'
    param['num_class'] = 58
    param['eta'] = 0.12
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['eval_metric'] = 'mlogloss'
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return model
    

# preds = np.zeros((test.shape[0], 1))
print('training')
model = runXGB(comments_train, train_l, comments_valid,valid_l)
print('predicting')
preds = model.predict(xgb.DMatrix(comments_test))
y_te = [np.argmax(pred) for pred in preds]
submit_df = pd.DataFrame({"itemid": test["itemid"], "Category": y_te})
submit_df.to_csv("submission_xgbst.csv", index=False)



