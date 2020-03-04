from keras.models import Sequential , Model ,load_model
from pyimagesearch import datasets
from pyimagesearch import models
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
# from PIL import Image

#---------------- setup ----------------
IMAGE_SIZE = (512,512)
food = "napa_cabbage_soup"
path_file = "food/"+food
namefile = "model_"+food

top_attribute = ["filename","top_total","top_noodle","top_veggie","top_meat"]
left_attribute = ["filename","left_total","left_liquid","left_veggie","left_meat"]
right_attribute = ["filename","right_total","right_liquid","right_veggie","right_meat"]

file_opject =  open("models/"+namefile+"_max.txt","r")
num_max = file_opject.read()
file_opject.close()

list_data_max =  num_max.split(",")

max_data = []
i = 0 
while(i < len(list_data_max) ):
    max_data.append(float(list_data_max[i+1]))
    i = i+2


# mlp = load_model('models/'+food+'.h5')

print("[INFO] loading food attributes... : " , end='')
data_frame,position = datasets.load_attribute(path_file)
# data_frame = pd.DataFrame(data_frame,columns=col)
print("[INFO] loading food images...")
data_images = datasets.load_namepic(data_frame["filename"] , path_file)
# data_images = data_images /  255.0
# print(num_max)
print(max_data)

if position == "top":
	attribute = top_attribute
elif position == "right":
	attribute = right_attribute
elif position == "left":
	attribute = left_attribute
print(data_frame)

nor = max_data[0]
invest = pd.DataFrame(max_data[1:]).T



def predict_model( weight= []  , im = "" , model_food = ""):
    global nor,invest
    # max_data = []
    # i = 0 
    # while(i < len(list_data_max) ):
    #     max_data.append(int(list_data_max[i+1]))
    #     i = i+2
    # print(max_data)
    # model = load_model("models/"+food+".h5")

    weight = weight / nor
    im = cv2.imread(im)
    im = cv2.resize(im,(512,512))
    im = np.asarray([im/255.0])
    weight = [[weight]]
    preds = model.predict([weight,im])
    preds = preds * invest
    del im,weight
    return preds


channel1_true = []
channel2_true = []
channel3_true = []

channel1_pred = []
channel2_pred = []
channel3_pred = []

predict_err1 = []
predict_err2 = []
predict_err3 = []

tolist = data_frame.index.tolist()
# print(tolist)
data_frame = data_frame.T
print(data_frame)
model = load_model("models/"+food+".h5")

for i in tolist:
    df = data_frame[i]
    print("records :" , i)
    # print(df)
    print(df[attribute[1]],data_images[i])
    # print(df[attribute[2:]].values.tolist())
    f_input = [df[attribute[1]],data_images[i]]
    f_true = df[attribute[2:]].values.tolist()
    # print(f_input[1].shape)
    print("f_true:",f_true)
    channel1_true.append(f_true[0])
    channel2_true.append(f_true[1])
    channel3_true.append(f_true[2])

    f_pred = predict_model(f_input[0],f_input[1],food)
    f_pred = f_pred.values.tolist()
    print(f_pred[0])

    channel1_pred.append(round(f_pred[0][0],2))
    channel2_pred.append(round(f_pred[0][1],2))
    channel3_pred.append(round(f_pred[0][2],2))
    
    predict_err1.append(round((f_true[0] - f_pred[0][0]),2))
    predict_err2.append(round((f_true[1] - f_pred[0][1]),2))
    predict_err3.append(round((f_true[2] - f_pred[0][2]),2))

    # print("f_pred:",f_pred[0])

ch1_error = mean_absolute_error(channel1_true , channel1_pred)
ch2_error = mean_absolute_error(channel2_true , channel2_pred)
ch3_error = mean_absolute_error(channel3_true , channel3_pred)

# print(channel1_true)
# print(channel1_pred)
# print(predict_err1)

plt.subplot(3, 1, 1)
plt.title(str(attribute[2])+" MAE :" + str(round(ch1_error , 2)))
plt.scatter(channel1_true, predict_err1, color='r')
plt.xlabel('Actual weight [g]')
plt.ylabel('Predict Error [g]')

plt.xlim(0,max(channel1_true))
plt.ylim(-1*(max(predict_err1)),1 *max(predict_err1))

plt.subplot(3, 1, 2)
plt.title(str(attribute[3])+" MAE :" + str(round(ch2_error,2)))
plt.scatter(channel2_true, predict_err2, color='g')
plt.xlabel('Actual weight [g]')
plt.ylabel('Predict Error [g]')

plt.xlim(0,max(channel2_true))
plt.ylim(-1*(max(predict_err2)), 1 * max(predict_err2))

plt.subplot(3, 1, 3)
plt.title(str(attribute[4])+" MAE :" + str(round(ch3_error,2)))
plt.scatter(channel3_true, predict_err3, color='b')
# plt.scatter(grades_range, boys_grades, color='g')
plt.xlabel('Actual weight [g]')
plt.ylabel('Predict Error [g]')

plt.xlim(0,max(channel3_true))
plt.ylim(-1*(max(predict_err3)),1*max(predict_err3))
plt.show()