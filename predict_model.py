from keras.models import Sequential , Model ,load_model
from keras.layers import Dense, Flatten, Input ,Dropout, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
import cv2
import pandas as pd
from PIL import Image

#---------------- setup ----------------
BATCH_SIZE = 5
MAX_EPOCH = 2000
IMAGE_SIZE = (512,512)

file_opject =  open("models/Fried Noodles_max.txt","r")
num_max = int(file_opject.read())
file_opject.close()


model = load_model('models/Fried Noodles.h5')
print(model.summary())
print("[INFO] predicting house prices...")
im = cv2.imread("IMG_20191003_162432.jpg")
im = cv2.resize(im,(IMAGE_SIZE))
im = np.array(im/255.0)
atrr = [[1.000000]]
im = np.array([im])
attr = pd.DataFrame(atrr)
test = [[0.727564]  ,[0.141026]  ,[0.131410]]
test = pd.DataFrame(test).T
preds = model.predict([attr,im])
score = model.evaluate([attr,im], test)

print("preds :", preds*num_max)
print("score :", score)





    
    