#! /usr/bin/python2
import cv2
import time
import sys
from pandas import DataFrame
import RPi.GPIO as GPIO
import sys
sys.path.insert(0,"c:/python37/lib/site-packages/")
import logging
import time
import random
import microgear.client as microgear
import pyrebase
########### keras #########3
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import glob
import itertools
import collections
from collections import defaultdict
import crop as crop
import predict_model as pred_weight
import predict_food as pred_class
import resize_to_netpie as resize
import predict_weight as cal

EMULATE_HX711=False
state = True
######### set switch ##########333


cap=cv2.VideoCapture(0)

referenceUnit = 1150
referenceUnit2 = 1000
referenceUnit3 = 1000
i=1
if not EMULATE_HX711:
    
    from hx711 import HX711
    
    
    
    
else:
    from emulated_hx711 import HX711
    # Set pin 10 to be an input pin and set initial value to be pulled low (off)

def cleanAndExit():
    print("Cleaning...")

    if not EMULATE_HX711:
        GPIO.cleanup()
        
    print("Bye!")
    sys.exit()

#######(dout,sdk)
hx = HX711(20,21) #sensor  mid
hx2 = HX711(6,13)  #sensor rigth
hx3 = HX711(19,26) #sensor lelf
#buttom_state = buttom(15)
buttonPin=22 #pin 15
GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)




hx.set_reading_format("MSB", "MSB")
hx2.set_reading_format("MSB", "MSB")
hx3.set_reading_format("MSB", "MSB")



################# Calibate ############
hx.set_reference_unit(referenceUnit)
hx2.set_reference_unit(referenceUnit2)
hx3.set_reference_unit(referenceUnit3)

hx.reset()
hx2.reset()
hx3.reset()

hx.tare()
hx2.tare()
hx3.tare()

print("Tare done! Add weight now...")


#print("Load cell")
#print("  lelf      center      rigth")
#########send date ############# 
def connection():
    logging.info("Now I am connected with netpie")

def subscription(topic,message):
    logging.info(topic+" "+message)

def disconnect():
    logging.debug("disconnect is work")
def convert_np_to_float(n = []):
    new_num = []
    for i in range(len(n)):
        new_num.append(float(n[i]))
    return new_num
##########crop scale
Pt1 = [(52, 20), (405, 239)] #กลาง
Pt2 = [(232, 256), (407, 453)] #ซ้าย
Pt3 = [(58, 258), (221, 454)] #ขวา

config = {
        "apiKey": "AIzaSyBmPnew7RwJDAu_c8oO0YdJ8bIvodJAaM4",
        "authDomain": "pinto-b2143.firebaseapp.com",
        "databaseURL": "https://pinto-b2143.firebaseio.com",
        "projectId": "pinto-b2143",
        "storageBucket": "pinto-b2143.appspot.com",
        "messagingSenderId": "660599466233",
        "appId": "1:660599466233:web:5de7405a3fb1771d1b5a28",
        "measurementId": "G-J9CJDEZ12G"
}

firebase = pyrebase.initialize_app(config)
# db = firebase.database()
store = firebase.storage()

print("INFO : Connecting Netpie !!!")
appid = "SendPic"
gearkey = "ZlLVTgx1SCXW1xJ"
gearsecret =  "E4gnN6yKtdG0DC8WWDDFnhR5q"
microgear.create(gearkey,gearsecret,appid,{'debugmode': True})

weight_list = []
#########3main ###########3
while True:
    ret,im =cap.read()
    cv2.rectangle(im, Pt1[0], Pt1[1], (0, 255, 0), 2)
    cv2.rectangle(im, Pt2[0], Pt2[1], (0, 255, 0), 2)
    cv2.rectangle(im, Pt3[0], Pt3[1], (0, 255, 0), 2)
    cv2.imshow("Food tray :", im)
    button_state = GPIO.input(buttonPin)
  
    val_mid = max(0,int(hx.get_weight(5)))
    val_rigth =max(0,int(hx2.get_weight(5)))
    val_lelf = max(0,int(hx3.get_weight(5)))
    
    #print(val_lelf,val_mid,val_rigth)    
    #print('buttonState',button_state)
    
    if cv2.waitKey(1) &button_state == 1:
    #if cv2.waitKey(1) & buttom_state == GPIO.HIGH :
        print('start system.....')
        
        sensor = {'sensor lelf':[val_lelf],
                  'sensor mid':[val_mid],
                  'sensor rigth':[val_rigth],}
        
        crop.crop_food(im)
        #weight_list.append(val_lelf+val_mid+val_rigth)
        #weight_list.append(val_lelf)
        #weight_list.append(val_rigth)
        #weight_list.append(val_mid)
        
        print("[INFO] calculator weight .. ")
        #weight_list = cal.weight(weight_list)
        print(val_lelf,val_mid,val_rigth)
        #######output img/
        ########### predict ###################
        print("[INFO] Split photo for predict food...")
        #class_C = pred_class.predict_food_class("img/center.jpg")
        #class_L = pred_class.predict_food_class("img/left.jpg")
        #class_R = pred_class.predict_food_class("img/right.jpg")

        #class_L = "Fried Noodles"
        #class_R = "Cucumber Soup"   

        class_C = "empty"
        class_L = "egg-soup" 
        class_R = "empty"

        print("[INFO] Split photo for predict model...")
        if( class_C != "empty"):
            C = pred_weight.predict_model(val_mid,"img/center.jpg",class_C)[0]
        else :
            C = [0,0,0]
        if( class_L != "empty"):
            L = pred_weight.predict_model(val_lelf,"img/left.jpg",class_L)[0]
            print(L)
        else :
            L = [0,0,0]
        if( class_R != "empty"):
            R = pred_weight.predict_model(val_rigth,"img/right.jpg",class_R)[0]
        else :
            R = [0,0,0]
            
        img_R = "img/right.jpg"
        img_C = "img/center.jpg" 
        img_L = "img/left.jpg"

        print("[INFO] Resize img to netpie... ")
        resize.resize_img("img/center.jpg","left")
        resize.resize_img("img/left.jpg","right")
        resize.resize_img("img/right.jpg","center")
        
        print("[INFO] Send data to netpie...")
        C = convert_np_to_float(C)
        L = convert_np_to_float(L)
        R = convert_np_to_float(R)
        
        P1 = round(L[0],2)  
        C1 = round(L[1],2)
        L1 = round(L[2],2)

        P2 = round(R[0],2)      
        C2 = round(R[1],2)
        L2 = round(R[2],2)

        P3 = round(C[0],2)      
        C3 = round(C[1],2)
        L3 = round(C[2],2)
        ############## send data ###############
        store.child("food/left.jpe").put('/home/pi/Downloads/HX711-master/HX711_Python3/hx711py/img/left.jpg')
        store.child("food/rigth.jpe").put("/home/pi/Downloads/HX711-master/HX711_Python3/hx711py/img/right.jpg")
        store.child("food/center.jpe").put("/home/pi/Downloads/HX711-master/HX711_Python3/hx711py/img/center.jpg")

        url_L = store.child("food/left.jpe").get_url(None)
        url_R = store.child("food/rigth.jpe").get_url(None)
        url_C = store.child("food/center.jpe").get_url(None)
        
        print("INFO : Upload Success !!!")
        microgear.setalias("a")
        microgear.on_connect = connection
        microgear.on_message = subscription
        microgear.subscribe("/mails")

        microgear.connect()
        L = url_L + "&token=28ef2706-806f-4699-afe1-ff737ac9b4da"
        R = url_R + "&token=d28ba572-312b-468e-a234-354b96d14319"
        C = url_C + "&token=0b0a32a9-1a00-43c0-a665-19058365ff14"
        
        food1 = str(P1) + "," + str(C1)+ "," + str(L1) +"," + str(L)+","+str(P2) + "," + str(C2)+ "," + str(L2) +"," + str(R)+","+str(P3) + "," + str(C3)+ "," + str(L3) +"," + str(C)
        microgear.chat("a",food1)
        time.sleep(10)
        microgear.on_disconnect = disconnect
        microgear.disconnect()
        weight_list = []
        print('Sent data Done!!!!!!!!!!!')
    
    #print(val_lelf,val_mid,val_rigth)
    hx.power_down()
    hx2.power_down()
    hx3.power_down()
        
    hx.power_up()
    hx2.power_up()
    hx3.power_up()
        
    time.sleep(0.1)
       
cap.release()
cv2.destroyAllWindow
        
                        
        


