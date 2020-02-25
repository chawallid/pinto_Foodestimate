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

top_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]
left_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]
right_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]


def write_data(line_ = 0 , file = "" , values = []):
	file_opject = open("models/"+file+"_max.txt","w")
# print(weight[0])
# print(data_frame[weight[0]].max()) 
	for i in range(line_):
		data_write =  top_attribute[i+1] +","+ str(values[top_attribute[i+1]].max()) 
		if(i >= line_ -1):
			file_opject.write(data_write )
		else :
			file_opject.write(data_write + ",")
	file_opject.close()	
	return 0

BATCH_SIZE = 5
MAX_EPOCH = 350
IMAGE_SIZE = (512,512)
food = "fried_noodles"
path_file = "food/fried_noodles"
namefile = "model_"+food

print("[INFO] loading food attributes...")
data_frame = datasets.load_attribute(path_file)
# data_frame = pd.DataFrame(data_frame,columns=col)
print("[INFO] loading food images...")
data_images = datasets.load_image(data_frame["filename"] , path_file)
data_images = data_images /  255.0

print("[INFO] Normolize working")
print(data_frame[top_attribute[1:]])
write_data(len(top_attribute[1:]),namefile,data_frame[top_attribute[1:]])
data_frame = data_frame[top_attribute[1:]] / data_frame[top_attribute[1:]].max()
print(data_frame[top_attribute[1:]])
# num_max = data_frame[col[len(col)-1]].max()
# num_max = float(num_max)
# file_opject = open("models/"+food+"_max.txt","w") 
# file_opject.write(str(num_max))
# file_opject.close()

print("[INFO] Normolize success !!! [OK]")
# data_frame  = data_frame[col[1:len(col)]].div(num_max)

print("[INFO] processing data...")
split = train_test_split(data_frame ,data_images ,test_size=0.2 , random_state=30)
(trainAttrX,testAttrX, trainImageX,testImageX) = split

trainAttrX = pd.DataFrame(trainAttrX , columns = top_attribute[1:])
testAttrX = pd.DataFrame(testAttrX , columns = top_attribute[1:])

#train
train_AttrX = trainAttrX[top_attribute[1]]
train_AttrY = trainAttrX[top_attribute[2:]]
#test
test_AttrX = testAttrX[top_attribute[1]]
test_AttrY = testAttrX[top_attribute[2:]]


mlp = models.create_mlp(1, regress=False)
cnn = models.create_cnn(IMAGE_SIZE[0], IMAGE_SIZE[1], 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(4, activation="relu")(combinedInput)
last_output = Dense(3, activation="linear")(x)

model = Model(inputs=[mlp.input,cnn.input], outputs=last_output)

model = load_model('models/'+food+'.h5')
opt = Adam(lr=1e-3, decay=1e-3 / 100)
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
