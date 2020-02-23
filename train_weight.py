from keras.models import Sequential , Model ,load_model
from keras.layers import Dense, Flatten, Input ,Dropout, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
import cv2
import pandas as pd

BATCH_SIZE = 10
MAX_EPOCH = 20
namefile = "model_weight"
path_file = "food"
weight = ["left_sensor","top_sensor","right_sensor","left_total","top_total","right_total"]

print("[INFO] loading weight attributes...")
data_frame = datasets.load_weight(path_file)
print(data_frame)


print("[INFO] Normolize working...")
normolize = preprocessing.MinMaxScaler(feature_range = (0,1))
data_frame = normolize.fit_transform(data_frame)
print(data_frame)
# data_frame = normolize.inverse_transform(data_frame)
# print(data_frame)

# num_max = data_frame[col[0]].max()
# num_max = float(num_max)
# file_opject = open("models/"+food+"_max.txt","w") 
# file_opject.write(str(num_max))
# file_opject.close()


print("[INFO] Normolize success [OK]")
# data_frame  = data_frame[col[0:len(col)]].div(num_max)
# print(data_frame)

print("[INFO] processing data...")
split = train_test_split(data_frame,test_size=0.25 , random_state=42)
(trainAttrX,testAttrX) = split


trainAttrX = pd.DataFrame(trainAttrX , columns = weight)
testAttrX = pd.DataFrame(testAttrX , columns = weight)
# print(trainAttrX)
#train
train_AttrX = trainAttrX[weight[:3]]
train_AttrY = trainAttrX[weight[3:]]
# print(train_AttrX)
# print(train_AttrY)
#test
test_AttrX = testAttrX[weight[:3]]
test_AttrY = testAttrX[weight[3:]]

print(test_AttrX)
print(test_AttrY)

mlp = models.create_mlp(3, regress=False)
mlp.add(Dense(8, activation="relu"))
mlp.add(Dense(16, activation="relu"))
mlp.add(Dense(32, activation="relu"))
mlp.add(Dense(64, activation="relu"))
mlp.add(Dense(64, activation="relu"))
mlp.add(Dense(32, activation="relu"))
mlp.add(Dense(16, activation="relu"))

mlp.add(Dense(8, activation="relu"))
mlp.add(Dense(4, activation="relu"))
mlp.add(Dense(3, activation="linear"))

# mlp = load_model('models/'+namefile+'.h5')
# opt = Adam(lr=1e-3, decay=1e-3 / 300)
mlp.compile(loss="mean_squared_error",
            metrics=['accuracy'],
            optimizer="adam")

print(mlp.summary())

print("[INFO] training model...")
checkpoint = ModelCheckpoint('models/'+namefile+'.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = mlp.fit(
    train_AttrX, train_AttrY,
	validation_data=(test_AttrX, test_AttrY),
	epochs=MAX_EPOCH,
	batch_size=BATCH_SIZE,
	callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.show()
# list_w = []
W = [[257,66,50]]
W = normolize.fit_transform(W)
print(W)
W = pd.DataFrame(W).T
print(W.shape)
preds = mlp.predict(W)
print(preds)