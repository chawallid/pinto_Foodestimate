from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pandas as pd
import itertools
import collections
from collections import defaultdict



BATCH_SIZE = 5
IMAGE_SIZE = (512,512)

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]

def predict_10_crop(img, ix, top_n=5, plot=False, preprocess=True):
    flipped_X = np.fliplr(img)
    crops = [
        img[:512,:512, :], # Upper Left
        img[:512, img.shape[1]-512:, :], # Upper Right
        img[img.shape[0]-512:, :512, :], # Lower Left
        img[img.shape[0]-512:, img.shape[1]-512:, :], # Lower Right
        center_crop(img, (512, 512)),
        
        flipped_X[:512,:512, :],
        flipped_X[:512, flipped_X.shape[1]-512:, :],
        flipped_X[flipped_X.shape[0]-512:, :512, :],
        flipped_X[flipped_X.shape[0]-512:, flipped_X.shape[1]-512:, :],
        center_crop(flipped_X, (512, 512))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    best_idx = collections.Counter(preds).most_common(1)[0][0]
    acc1 = y_pred[0][best_idx]    
    
    return preds , acc1

def predict_image(img):
    preds , acc = predict_10_crop(np.array(img), 0)
    best_pred = collections.Counter(preds).most_common(1)[0][0]
    print(classname[best_pred],"Accuracy: %.2f%%" % (acc*100))

datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    'food/test',
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=15,
    color_mode='rgb',
    class_mode='categorical')

#Test Model
model = load_model('class_food.h5')

file_opject =  open("class_name.txt","r")
classname = file_opject.read()
file_opject.close()
classname = classname.split(",")
print("classname :",classname[0])

path  = '*.jpg'
list_img = []
for img in glob.glob(path):
    pic = cv2.imread(img , cv2.IMREAD_UNCHANGED)
    pic = cv2.resize(pic,(512,512))
    print("IMG : " , img)
    predict_image(pic)
