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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



BATCH_SIZE = 5
MAX_EPOCH = 200
IMAGE_SIZE = (512,512)
food = "Cucumber Soup"
path_file = "crop/2019-10-07 Cucumber Soup"
print("[INFO] loading food attributes...")
data_frame,col = datasets.load_attribute(path_file)
data_frame = pd.DataFrame(data_frame,columns=col)

print("[INFO] loading food images...")
data_images = datasets.load_image(data_frame["filename"] , path_file)
data_images = data_images /  255.0

print("[INFO] Normolize working")
num_max = data_frame[col[len(col)-1]].max()
num_max = float(num_max)
file_opject = open("models/"+food+"_max.txt","w") 
file_opject.write(str(num_max))
file_opject.close()


print("[INFO] Normolize success")
data_frame  = data_frame[col[1:len(col)]].div(num_max)

print("[INFO] processing data...")
split = train_test_split(data_frame ,data_images ,test_size=0.25 , random_state=42)
(trainAttrX,testAttrX, trainImageX,testImageX) = split

#train
train_AttrX = trainAttrX[col[len(col)-1]]
train_AttrY = trainAttrX[col[1:len(col)-1]]
#test
test_AttrX = testAttrX[col[len(col)-1]]
test_AttrY = testAttrX[col[1:len(col)-1]]


mlp = models.create_mlp(1, regress=False)
cnn = models.create_cnn(IMAGE_SIZE[0], IMAGE_SIZE[0], 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(4, activation="relu")(combinedInput)
last_output = Dense(3, activation="linear")(x)

model = Model(inputs=[mlp.input,cnn.input], outputs=last_output)

# model = load_model('2019-10-04 Fried Noodles.h5')
opt = Adam(lr=1e-3, decay=1e-3 / 300)
model.compile(loss="mean_squared_error",
            metrics=['accuracy'],
            optimizer=opt)

print(model.summary())

print("[INFO] training model...")
checkpoint = ModelCheckpoint('models/'+food+'.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model.fit(
    [train_AttrX,trainImageX], train_AttrY,
	validation_data=([test_AttrX,testImageX], test_AttrY),
	epochs=MAX_EPOCH,
	batch_size=BATCH_SIZE,
	callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.show()
