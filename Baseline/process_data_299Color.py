
# coding: utf-8

# In[1]:


import numpy as np
import glob
import cv2


# In[ ]:


data_dir='Data/*/*'
data_list=glob.glob(data_dir)
data=np.empty([1,299,299,3])
label=np.empty([1,])
for num, photo in enumerate(data_list):
    print (num, photo)
    photo_r=cv2.imread(photo)
    #print (photo_r)
    cropped=photo_r[0:360,140:500]
    cropped=cv2.resize(cropped,(299,299))
    cropped=cropped.reshape(1,299,299,3)
    data=np.append(data,cropped,axis=0)
    if 'Improper' in photo:
        label=np.append(label,np.array([1]),axis=0)
    else:
        label=np.append(label,np.array([0]),axis=0)

np.random.seed(42)
data=data[1:-1]
label=label[1:-1]
index=np.arange(len(label))
np.random.shuffle(index)
print (index)
data=data[index]
label=label[index]

np.savez_compressed('./data.npz', arr_0=data)
np.savez_compressed('./label.npz', arr_0=label)
