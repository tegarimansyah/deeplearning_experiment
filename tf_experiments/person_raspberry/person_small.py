
# coding: utf-8

# In[1]:

'''
Tegar Imansyah
Lenet with tensorflow and keras

ref:
1. https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
2. http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
'''

from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.preprocessing.image as image
import numpy as np
import matplotlib.pyplot as plt
import time


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 12810+8303
nb_validation_samples = 6319+4639
epochs = 50
batch_size = 16
input_shape = (img_width, img_height, 3) # 3 Channel, di belakang karena menggunakan tensorflow's channel_last



# In[3]:

print("[INFO] drawing networks...")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.summary()
model.load_weights('model/person_small_1491976587441.h5')

def check(probs,answer):
    if(probs[0,0] == 0):
        print('Predict: No Victim, Actual: ' + answer)
    else:
        print('Predict: Victim, Actual: ' + answer)
        
def showtest():  
    import os, random
    for i in range(9):
        if (np.random.randint(1,2+1) == 1):
            chosen_file = random.choice(os.listdir("data/validation/pos"))
            img_path = 'data/validation/pos/' 
            img_path+= str(chosen_file)
            answer = 'Victim'
        else:
            chosen_file = random.choice(os.listdir("data/validation/neg"))
            img_path = 'data/validation/neg/' 
            img_path+= str(chosen_file)
            answer = 'No Victim'

        
        img = image.load_img(img_path, target_size=(150,150))
        # plt.imshow(img)
        # plt.show()
        print(img_path)
        img = image.img_to_array(img)
        img = img.reshape((1,) + img.shape)
        start_time = time.time()
        probs = model.predict(img)
        check(probs, answer)
        print('Predicted in %s seconds' % (time.time()-start_time))


showtest()
