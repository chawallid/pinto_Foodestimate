from keras.models import Sequential , Model ,load_model
import numpy as np
import cv2
import pandas as pd
from PIL import Image

#---------------- setup ----------------
IMAGE_SIZE = (512,512)

def predict_model( weight=0 , img = "" , model_food = ""):
    print("weight",weight)
    file_opject =  open("models/"+model_food+"_max.txt","r")
    num_max = float(file_opject.read())
    file_opject.close()
    weight = float(weight / num_max)
    # print("weight",weight)
    model = load_model("models/"+model_food+".h5")
    im = cv2.imread(img)
    im = cv2.resize(im,(512,512))
    im = np.array(im/255.0)
    # cv2.imshow(im)
    atrr = [[weight]]
    im = np.array([im])
    attr = pd.DataFrame(atrr)
    preds = model.predict([attr,im])
    print("preds",preds)
    preds[0][0] = preds[0][0] / 2
    preds[0][1] = preds[0][1] / 4
    preds[0][2] = preds[0][2] / 4
    
    return(preds)





    
    