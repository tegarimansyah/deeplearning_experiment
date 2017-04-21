
# coding: utf-8

# In[1]:

# $ ls validation/neg/ | wc -l
# 6319
# $ ls validation/pos/ | wc -l
# 4639
# $ ls train/pos/ | wc -l
# 8303
# $ ls train/neg/ | wc -l
# 12810

from keras import applications # Load VGG
# from keras import optimizers # Compile
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.preprocessing.image as image
import numpy as np
import time
# import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 150, 150

# train_data_dir = 'data/train'
# validation_data_dir = 'data/validation'
# nb_train_samples = 12810+8303
# nb_validation_samples = 6319+4639
# epochs = 50
# batch_size = 16
input_shape = (img_width, img_height, 3) # 3 Channel, di belakang karena menggunakan tensorflow's channel_last


# path to the model weights files.
# weights_path = '../keras/examples/vgg16_weights.h5'
# top_model_weights_path = 'fc_model.h5'


# In[2]:

# prepare data augmentation configuration
# train_datagen = image.ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = image.ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')


# In[3]:

# build the VGG16 network
base_model = applications.VGG16(weights = None, include_top= False, input_shape=(150, 150, 3))
print('Model loaded.')


# In[4]:

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model = Model(inputs=base_model.input, outputs=top_model(base_model.output)) # Net =  VGG.input -> VGG.output -> top_model -> output


# In[10]:

# Change Trainable layers until layer 15

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.load_weights('model/person1491819477126.h5')

# Trainable = true
# for layer in model2.layers:
#     layer.trainable = True

# Trainable = false for 0 - 14
for layer in model.layers[:25]:
    layer.trainable = False

# Show All    
i = 0
for layer in model.layers:
    print(i,layer,layer.trainable)
    i+=1

# model2.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

model.summary()


# In[12]:

# # fine-tune the model
# model2.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

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


# # Main Program Sampai Disini
