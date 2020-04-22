from keras.models import load_model
import cv2
import pandas as pd

#name file model
namefile = "model_weight"

#column weight for fit MLP
weight = ["left_sensor","top_sensor","right_sensor","left_total","top_total","right_total"]



def predicts_weight_from_loadcell(W = []):
    global namefile
    #list num for normalize
    max_data = []

    #read invest-normalized from file
    file_opject =  open("models/"+namefile+"_max.txt","r")
    num_max = file_opject.read()
    file_opject.close()
    list_data_max =  num_max.split(",")
    i = 0 
    while(i < len(list_data_max) ):
        max_data.append(float(list_data_max[i+1]))
        i = i+2

    #change W to dataframe 
    W = pd.DataFrame(W)

    #normalize for predict
    normal = pd.DataFrame(max_data[:3])

    #invest-normalize from predict
    invest = pd.DataFrame(max_data[3:])

    #calculater W 
    W = ( W / normal )

    mlp = load_model("models/"+namefile+".h5")

    #predict weight 
    print(W)
    preds = mlp.predict(W.T)

    #return values and invest to true values
    return preds * invest.T


print(predicts_weight_from_loadcell([47,37,0]))



    
