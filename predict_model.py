from keras.models import load_model
import numpy as np
import cv2
import pandas as pd

def predict_model( weight= []  , im = "" , model_food = ""):
    global nor,invest

    file_opject =  open("models/"+model_food+"_max.txt","r")
    num_max = file_opject.read()
    file_opject.close()
    list_data_max =  num_max.split(",")

    max_data = []
    i = 0 
    while(i < len(list_data_max) ):
        max_data.append(int(list_data_max[i+1]))
        i = i+2

    model = load_model("models/"+model_food+".h5")
        
    nor = max_data[0]
    invest = pd.DataFrame(max_data[1:]).T

    weight = weight / nor
    im = cv2.imread(im)
    im = cv2.resize(im,(512,512))
    im = np.asarray([im/255.0])
    weight = [[weight]]
    preds = model.predict([weight,im])
    preds = preds * invest
    del im,weight
    return preds


