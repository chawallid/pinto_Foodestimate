from keras.models import Sequential , Model ,load_model
from sklearn.metrics import mean_absolute_error
from pyimagesearch import datasets
from pyimagesearch import models
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
import cv2
import pandas as pd

namefile = "model_weight"
path_file = "food"
weight = ["left_sensor","top_sensor","right_sensor","left_total","top_total","right_total"]

file_opject =  open("models/"+namefile+"_max.txt","r")
num_max = file_opject.read()
file_opject.close()
print(num_max)
list_data_max =  num_max.split(",")

max_data = []
i = 0 
while(i < len(list_data_max) ):
    max_data.append(int(list_data_max[i+1]))
    i = i+2
print(max_data)


mlp = load_model('models/'+namefile+'.h5')
# W = [115,329,22]

def predicts_weight_from_loadcell(W = []):
    W = pd.DataFrame(W)
    normal = pd.DataFrame(max_data[:3])
    invest = pd.DataFrame(max_data[3:])

    W = ( W / normal ) 
    preds = mlp.predict(W.T)
    # print(preds * invest.T)

    return preds * invest.T



print("[INFO] loading weight attributes...")
data_frame = datasets.load_weight(path_file)

# for i in range(len(data_frame.index)):
#     print(data_frame.T[i])
left_true = []
top_true = []
right_true = []
left_pred = []
top_pred = []
right_pred = []
predict_err1 = []
predict_err2 = []
predict_err3 = []


print(data_frame.loc[data_frame['left_total'] == data_frame["left_total"].min()])
data_frame = data_frame.loc[data_frame['left_total'] == data_frame["left_total"].min()]
list_index = data_frame.index.tolist()
num  = 0
for i in list_index :
    W_pred = [data_frame.T[i].T[weight[:3]][0],data_frame.T[i].T[weight[:3]][1],data_frame.T[i].T[weight[:3]][2]]
    W_true = [data_frame.T[i].T[weight[3:]][0],data_frame.T[i].T[weight[3:]][1],data_frame.T[i].T[weight[3:]][2]]
    pre_W = W_pred.copy()
    W_pred = predicts_weight_from_loadcell(W_pred)
    
    left_pred.append(W_pred.values.tolist()[0][0])
    predict_err1.append(W_pred.values.tolist()[0][0] - W_true[0])
    left_true.append(W_true[0])
    # print(i)
    if((W_pred.values.tolist()[0][0] - W_true[0]) > 5.0):
        num += 1
        print(i,":",pre_W,W_true,round(W_pred.values.tolist()[0][0],2),round(W_pred.values.tolist()[0][1],2),round(W_pred.values.tolist()[0][2],2))

    # top_pred.append(W_pred.values.tolist()[0][1])
    # predict_err2.append(W_pred.values.tolist()[0][1] - W_true[1])
    # top_true.append(W_true[1])
    # right_pred.append(W_pred.values.tolist()[0][2])
    # predict_err3.append(W_pred.values.tolist()[0][2] - W_true[2])
    # right_true.append(W_true[2])
    
    
print("data_error :",num)
left_error = mean_absolute_error(left_true, left_pred)
# top_error = mean_absolute_error(top_true , top_pred)
# right_error = mean_absolute_error(right_true,right_pred)
# print(left_true)
# print(predict_err1)
plt.subplot(3, 1, 1)
plt.title("Left MAE :" + str(round(left_error , 2)))
plt.scatter(left_true, predict_err1, color='r')
plt.xlabel('Actual weight [g]')
plt.ylabel('Predict Error [g]')

plt.xlim(0,max(left_true))
plt.ylim(-1*(max(predict_err1)),max(predict_err1))

# plt.subplot(3, 1, 2)
# plt.title("Top MAE :" + str(top_error))
# plt.scatter(top_true, predict_err2, color='g')
# plt.xlabel('Actual weight [g]')
# plt.ylabel('Predict Error [g]')

# plt.xlim(0,max(top_true))
# plt.ylim(-1*(max(predict_err2)),max(predict_err2))

# plt.subplot(3, 1, 3)
# plt.title("Right MAE :" + str(right_error))
# plt.scatter(right_true, predict_err3, color='b')
# # plt.scatter(grades_range, boys_grades, color='g')
# plt.xlabel('Actual weight [g]')
# plt.ylabel('Predict Error [g]')

# plt.xlim(0,max(right_true))
# plt.ylim(-1*(max(predict_err3)),max(predict_err3))
plt.show()



# print("left_error :", left_error)
# print("top_error :" , top_error)
# print("right_error :", right_error)
    
