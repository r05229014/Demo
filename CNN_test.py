from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

model = load_model('./models/model_CNN.h5')

img = cv2.imread('./real_img/35062719_1941302955894222_4729918758619971584_n.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turn RGB to gray
img = cv2.resize(img, (28,28)) # resize
#plt.imshow(img, 'gray')
#plt.show()

img = img.astype('float32')
img /= 255 # normalize
img = img.reshape(1,28,28,1) #　(batch_size, width, height, channel)

pre = model.predict(img)
pre = np.argmax(pre)
print(pre)
