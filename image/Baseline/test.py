import numpy as np
import glob
import cv2
from keras.models import load_model

testdata_dir='../Test/*/*'
list=glob.glob(testdata_dir)
answer=np.empty([0])
prediction=np.empty([0])
model=load_model('weights/NASNET224Cweights.08-0.53.hdf5')


for num, photo in enumerate(list):
    print (num, photo)
    if 'Improper' in photo:
        answer=np.append(answer,1)
        print(' Answer:Improper')
    else:
        answer=np.append(answer,0)
        print(' Answer:Proper')
    pred=np.random.randint(0,2)
    #improper
    photo_r=cv2.imread(photo)
    #print (photo_r)
    cropped=photo_r[0:360,140:500]
    cropped=cv2.resize(cropped,(224,224))
    cropped=cropped.reshape(1,224,224,3)
    #cropped=np.multiply(cropped , 1.0/255.0)
    pred=model.predict(cropped)
    pred=np.argmax(pred,axis=-1)
    if pred == 1:
        prediction=np.append(prediction,1)
        print(' Pred:Improper')
    else:
        prediction=np.append(prediction,0)
        print(' Pred:Proper')

correct=0

for i in range(0,len(answer)):
    if answer[i]==prediction[i]:
        correct=correct+1
print (answer)
print (prediction)
Accuracy= correct/len(answer)
print('The accuracy is ',Accuracy)
