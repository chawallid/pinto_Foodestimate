from keras.models import Sequential , Model ,load_model
from keras.layers import Dense, Flatten, Input ,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import pandas as pd
from keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyimagesearch import datasets
from pyimagesearch import models
import resize_to_netpie as resize
import predict_model as pred_weight
import predict_food as pred_class
import crop as crp

BATCH_SIZE = 5
MAX_EPOCH = 200
IMAGE_SIZE = (512,512)
weight_C = 0
weight_L = 40
weight_R = 93
def convert_np_to_float(n = []):
    new_num = []
    for i in range(len(n)):
        new_num.append(float(n[i]))
    return new_num

def main_function():
    # img = "2019_11_29-ancient_pork-set_1-1.jpg"
    print("[INFO] Take photo...")

    print("[INFO] Crop image follow position...")
    # crp.crop_food("template.jpg")

    print("[INFO] Split photo for predict food...")
    class_C = pred_class.predict_food_class("img/center.jpg")
    class_L = pred_class.predict_food_class("img/left.jpg")
    class_R = pred_class.predict_food_class("img/right.jpg")

    print("[INFO] Split photo for predict model...")
    C = pred_weight.predict_model(weight_C,"img/center.jpg",class_C)[0]
    img_C = "img/center.jpg"
    L = pred_weight.predict_model(weight_L,"img/left.jpg",class_L)[0]
    img_L = "img/left.jpg"
    R = pred_weight.predict_model(weight_R,"img/left.jpg",class_R)[0]
    img_R = "img/right.jpg"

    print("[INFO] Resize img to netpie... ")
    resize.resize_img("img/center.jpg","left")
    resize.resize_img("img/left.jpg","right")
    resize.resize_img("img/right.jpg","center")

    print("[INFO] Send data to netpie...")
    C = convert_np_to_float(C)
    L = convert_np_to_float(L)
    R = convert_np_to_float(R)
    
    return C,img_C,L,img_L,R,img_R

main_function()