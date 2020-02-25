from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import csv

weight = ["left_sensor","top_sensor","right_sensor","left_total","top_total","right_total"]
top_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]
left_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]
right_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]

def load_attribute(_path):
    path = _path+".csv"
    read = pd.read_csv(path)
    print(read["position"][0])
    if( read["position"][0] == "top"):
        df =  read[top_attribute]
    elif(read["position"][0] == "right"):
        pass
    elif(read["position"][0] == "left"):
        pass
    # df = read
    return df

def load_image(df , path ):
    images = [] 
    for i in df:
        outputImage = np.zeros((512, 512, 3), dtype="uint8")
        base_Path = path + "/" + i
        image = cv2.imread(base_Path)
        image = cv2.resize(image, (512, 512))
        outputImage[0:512,0:512] = image
        images.append(outputImage)
            
    return np.array(images)

def load_weight(_path):
    list_dataset = []
    for i in glob.glob(_path+"/*.csv"):
        list_dataset.append(i)

    data = pd.DataFrame(columns=weight)

    # print(list_dataset[1])
    for i in list_dataset :
        print("load_weight from :", str(i))
        read_data = pd.read_csv(str(i))
        read_data = pd.DataFrame(read_data[weight], columns = weight)
        data = data.append(read_data , ignore_index=True)

    return data


    
