from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.applications.vgg16 import preprocess_input
import numpy as np

img_path = 'preview/cat_0_290.jpeg'
img = image.load_img(img_path, target_size=(150,150))

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
