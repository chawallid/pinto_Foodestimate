from keras.models import Sequential , Model ,load_model
from keras.layers import Dense, Flatten, Input ,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

#---------------- setup ----------------
BATCH_SIZE = 5
MAX_EPOCH = 200
IMAGE_SIZE = (512,512)
# TRAIN_IM = 160
# VALIDATE_IM = 15

#---------------- model CNN ----------------
# model = Sequential()
model = load_model('class_food.h5')

input = Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3))
conv1 = Conv2D(8,3,activation='relu', padding="same" , kernel_initializer='he_normal')(input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(16,3,activation='relu', padding="same", kernel_initializer='he_normal')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(32,3,activation='relu', padding="same", kernel_initializer='he_normal')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(64,3,activation='relu', padding="same", kernel_initializer='he_normal')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(128,3,activation='relu', padding="same", kernel_initializer='he_normal')(pool4)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

conv6 = Conv2D(256,3,activation='relu', padding="same", kernel_initializer='he_normal')(pool5)
pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

conv7 = Conv2D(512,3,activation='relu', padding="same", kernel_initializer='he_normal')(pool6)
pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

flat = Flatten()(pool7)
hidden = Dense(64, activation='relu')(flat)
hidden = Dense(64, activation='relu')(hidden)

output = Dense(30, activation='softmax')(hidden)
model = Model(inputs=input, outputs=output)
    
model.compile(  loss='binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy']      )

print(model.summary())

#---------------- datagenarator ----------------

datagen = ImageDataGenerator(rescale=1./255)

train_food = datagen.flow_from_directory(
    'food/train',
    class_mode='categorical',
    color_mode='rgb',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True)

validation_food = datagen.flow_from_directory(
    'food/validation',
    class_mode='categorical',
    color_mode='rgb',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_food = datagen.flow_from_directory(
    'food/test',
    class_mode='categorical',
    color_mode='rgb',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True)


checkpoint = ModelCheckpoint('class_food.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model.fit_generator(
    train_food,
    epochs=MAX_EPOCH,
    steps_per_epoch=len(train_food),
    validation_data=validation_food,
    validation_steps=len(validation_food),
    callbacks=[checkpoint])



plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.show()