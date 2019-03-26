import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,MaxPool2D,Conv2D,GlobalAveragePooling2D,BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
import pandas as pd
import csv
import os
from keras import applications
from keras import metrics
import keras.backend as K
import shutil
import argparse



BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 10
BASE_DIR = '/Users/i502640/Downloads/train/test'
# '/home/students/student3_2a/train'
IMAGE_WIDTH, IMG_HEIGHT = 299, 299







def add_new_layer(base_model):
    x = base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.1)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(input = base_model.input, output=predictions)
    return model

def set_up_to_finetune(base_model,model):
    for layer in base_model.layers:
        layer.trainable = False
#     for layer in base_model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
#         layer.trainable = True


# def plot_confusion_matrix(cm, classes, savefig,
#                         normalize=False,                       
#                         cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     import itertools
#     plt.switch_backend('agg')
#     matplotlib.rcParams.update({'font.size': 6})
#     plt.figure(figsize=(20,20))
#     plt.figure()
#     plt.imshow(cm, cmap=cmap)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig(savefig)

def train():
    ### data preprocessing


    train_datagen = ImageDataGenerator(
    validation_split =0.2)
    train_generator = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMG_HEIGHT,IMAGE_WIDTH),
        # shuffle=true,
        class_mode = 'categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMG_HEIGHT,IMAGE_WIDTH),
        # shuffle=true,
        class_mode='categorical',
        subset='validation')

    base_model = applications.VGG19(input_shape=(299,299,3),weights='imagenet', include_top=False)

    model = add_new_layer(base_model)
    set_up_to_finetune(base_model,model)
    # parallel_model = multi_gpu_model(model, gpus=4)


    model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


    checkpoint = ModelCheckpoint('InceptionV3(trainale).h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


    history_fit = model.fit_generator(
        train_generator,
        epochs = EPOCHS,
        verbose = 1,
        steps_per_epoch = 50,
        validation_data = validation_generator,
        validation_steps=10,
#        class_weight = 'auto',
        callbacks = [checkpoint,early]
)
    model.save('InceptionV3(trainale).h5')






if __name__ =='__main__':
    train()




# with open('train.csv','r') as csvfile:
#   reader = csv.DictReader(csvfile)
#   for cat in range(0,59):
#       path = os.path.join(BASE_DIR,str(cat))
#       if(not os.path.exists(path)):
#           os.mkdir(path)
#   for row in reader:
#       if os.path.exists(os.path.join(BASE_DIR,row['Category'])):
#           if(not os.path.exists(row['image_path'])):
#               print(row['image_path'])

#           else:
#               print('aaaa',row['image_path'])
#               shutil.copy(os.path.join(row['image_path']), os.path.join(BASE_DIR,row['Category']))
