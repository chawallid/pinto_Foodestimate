from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import csv

def load_attribute(_path):
    for i in glob.glob(_path+"/*.csv"):
        _path  = i
    df = []
    cols = []
    with open(_path) as csvfile:
        reader = csv.reader(csvfile)
        count = 0 
        col = 0
        keep = []
        for row in reader :
            if(count == 0):
                col = len(row)
                count += 1
            keep.append(row)
      
        check = 0 
        for row in keep:
            tmp = []
            if(check == 0 ):
                cols = row
                check +=1
                continue
            else:
                for i in range(col):
                    if(i != 0):
                        tmp.append(float(row[i]))
                    else:
                        tmp.append(row[i])       
            df.append(tmp)      
    return df,cols

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

   
    weight = ["left_sensor","top_sensor","right_sensor","left_total","top_total","right_total"]
    data = pd.DataFrame(columns=weight)

    # print(list_dataset[1])
    for i in list_dataset :
        print("load_weight from :", str(i))
        read_data = pd.read_csv(str(i))
        read_data = pd.DataFrame(read_data[weight], columns = weight)
        data = data.append(read_data , ignore_index=True)

    return data


    
