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
# path_file = "crop/2019-10-04 Fried Noodles"
path_file = "crop/2019-10-04 Fried Noodles"

print("[INFO] loading food attributes...")
data_frame,col = datasets.load_attribute(path_file)
data_frame = pd.DataFrame(data_frame,columns=col)
# print(data_frame["filename"])
print("[INFO] loading food images...")
data_images = datasets.load_image(data_frame["filename"] , path_file)
data_images = data_images /  255.0

print("[INFO] Normolize working")
num_max = data_frame[col[len(col)-1]].max()
num_max = float(num_max)

print("[INFO] Normolize success")
data_frame  = data_frame[col[1:len(col)]].div(num_max)
print("data_frame :" , data_frame)

print("[INFO] processing data...")
split = train_test_split(data_frame ,data_images ,test_size=0.25 , random_state=42)
(trainAttrX,testAttrX, trainImageX,testImageX) = split

#train
train_AttrX = trainAttrX[col[len(col)-1]]
train_AttrX = pd.DataFrame(train_AttrX)
train_AttrY = trainAttrX[col[1:len(col)-1]]
#test
test_AttrX = testAttrX[col[len(col)-1]]
test_AttrX = pd.DataFrame(test_AttrX)
test_AttrY = testAttrX[col[1:len(col)-1]]


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
# print(pd.DataFrame(test).T)
preds = model.predict([attr,im])
score = model.evaluate([attr,im], test)

print("preds :", preds)
print("score :", score)

# cv2.imshow('camera',im)
# cv2.waitKey(1)



    
    