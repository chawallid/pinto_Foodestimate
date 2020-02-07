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

food = "data_set_loadcell"

file_opject =  open("models/"+food+"_max.txt","r")
num_max = float(file_opject.read())
file_opject.close()

mlp = load_model('models/'+food+'.h5')

W = [254,69,61,124]
W = pd.DataFrame(W).T
print(W.shape)
W = W.div(num_max)
print(W)
preds = mlp.predict(W)
print(preds * num_max)