import os
#import h5py
import numpy as np
from keras.models import Sequential, Model
import tensorflow as tf
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19

############### LOAD DATA ###################
img_width, img_height = 299,299
bat_size = 10
nb_epoch = 50

#load data
x=np.load('data.npz')['arr_0']
y=np.load('label.npz')['arr_0']

#turn y to 2 cats and normalize x.
y_label = np_utils.to_categorical(y, num_classes=2)
x_train=x

################# CREATE MODEL ##################
base_model = VGG19(input_shape=(299,299,3),weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x=GlobalAveragePooling2D()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

learning_rate = 0.1
decay_rate = learning_rate/nb_epoch
momentum = 0.8
model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


################ TRAIN ##################

model_path='../weights/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint=ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
checklist=[checkpoint]

model.fit(x=x_train, y=y_label,epochs=nb_epoch, verbose=1,validation_split=0.2,batch_size=bat_size,callbacks=checklist)
