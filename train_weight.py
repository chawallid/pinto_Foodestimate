from keras.models import Sequential , Model ,load_model
from keras.layers import Dense, Flatten, Input ,Dropout, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
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
MAX_EPOCH = 200
food = "data_set_loadcell"
path_file = "weight/"
print("[INFO] loading food attributes...")
data_frame,col = datasets.load_weight(path_file)
data_frame = pd.DataFrame(data_frame,columns=col)
# print(data_frame)


print("[INFO] Normolize working")
num_max = data_frame[col[0]].max()
num_max = float(num_max)
file_opject = open("models/"+food+"_max.txt","w") 
file_opject.write(str(num_max))
file_opject.close()


print("[INFO] Normolize success")
data_frame  = data_frame[col[0:len(col)]].div(num_max)
print(data_frame)

print("[INFO] processing data...")
split = train_test_split(data_frame,test_size=0.25 , random_state=42)
(trainAttrX,testAttrX) = split

#train
train_AttrX = trainAttrX[col[4:]]
train_AttrY = trainAttrX[col[:4]]
# #test
test_AttrX = testAttrX[col[4:]]
test_AttrY = testAttrX[col[0:4]]

print(test_AttrX)
print(test_AttrY)

mlp = models.create_mlp(4, regress=False)
mlp.add(Dense(8, activation="relu"))
# mlp.add(Dense(16, activation="relu"))
# mlp.add(Dense(16, activation="relu"))
mlp.add(Dense(8, activation="relu"))
mlp.add(Dense(4, activation="linear"))

# # model = Model(inputs=[mlp.input,cnn.input], outputs=last_output)

# mlp = load_model('models/'+food+'.h5')
opt = Adam(lr=1e-3, decay=1e-3 / 300)
mlp.compile(loss="mean_squared_error",
            metrics=['accuracy'],
            optimizer=opt)

print(mlp.summary())
# 
print("[INFO] training model...")
checkpoint = ModelCheckpoint('models/'+food+'.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

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
# W = [257,66,50,141]
# W = pd.DataFrame(W).T
# print(W.shape)
# W = W.div(num_max)
# print(W)
# preds = mlp.predict(W)
# print(preds * num_max)