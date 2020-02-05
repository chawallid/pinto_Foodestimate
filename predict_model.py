from keras.models import Sequential , Model ,load_model
import numpy as np
import cv2
import pandas as pd
from PIL import Image

#---------------- setup ----------------
IMAGE_SIZE = (512,512)

def predict_model( weight=0 , img = "" , model_food = ""):

    file_opject =  open("models/"+model_food+"_max.txt","r")
    num_max = int(file_opject.read())
    file_opject.close()
    weight = float(weight / num_max)
    model = load_model("models/"+model_food+".h5")
    im = cv2.imread(img)
    im = cv2.resize(im,(512,512))
    im = np.array(im/255.0)
    atrr = [[weight]]
    im = np.array([im])
    attr = pd.DataFrame(atrr)
    preds = model.predict([attr,im])
    
    return(preds*num_max)





    
    