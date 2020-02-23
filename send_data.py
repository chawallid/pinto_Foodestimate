import sys
sys.path.insert(0,"c:/python37/lib/site-packages/")
import logging
import time
import random
import microgear.client as microgear
import pyrebase
import main as pinto
import cv2

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

#call main function
C,img_C,L,img_L,R,img_R,Loadcell_w,neuron_w,food_c,food_l,food_r = pinto.main_function()
# print("L :" , type(L[0]))
P1 = round(L[0],2)	
C1 = round(L[1],2)
L1 = round(L[2],2)

P2 = round(R[0],2)		
C2 = round(R[1],2)
L2 = round(R[2],2)

P3 = round(C[0],2)		
C3 = round(C[1],2)
L3 = round(C[2],2)

store.child("food/left.jpe").put(img_L)
store.child("food/rigth.jpe").put(img_R)
store.child("food/center.jpe").put(img_C)

url_L = store.child("food/left.jpe").get_url(None)
url_R = store.child("food/rigth.jpe").get_url(None)
url_C = store.child("food/center.jpe").get_url(None)
print("INFO : Upload Success !!!")


appid = "SendPic"
gearkey = "5kiUuISZVKUTAQX"
gearsecret =  "hHo1bpmB02SIMl1h5uRXhRepB"
microgear.create(gearkey,gearsecret,appid,{'debugmode': True})

print("INFO : Connecting Netpie !!!")

def connection():
    logging.info("Now I am connected with netpie")

def subscription(topic,message):
    logging.info(topic+" "+message)

def disconnect():
    logging.debug("disconnect is work")

microgear.setalias("detail")
microgear.setalias("namefood")
microgear.setalias("cal")
microgear.on_connect = connection
microgear.on_message = subscription
microgear.subscribe("/mails")
microgear.connect()

while True :
    print(time.clock())
    image = cv2.imread("right.jpg",cv2.COLOR_BAYER_BG2BGR)
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # microgear.setalias("a")
        # microgear.on_connect = connection
        # microgear.on_message = subscription
        # microgear.subscribe("/mails")
        # microgear.connect()

        #token_firebase
        L = url_L + "&token=28ef2706-806f-4699-afe1-ff737ac9b4da"
        R = url_R + "&token=d28ba572-312b-468e-a234-354b96d14319"
        C = url_C + "&token=0b0a32a9-1a00-43c0-a665-19058365ff14"

        # while True:
            # if (microgear.connect()):
        food1 = str(P1) + "," + str(C1)+ "," + str(L1) +"," + str(L)+","+str(P2) + "," + str(C2)+ "," + str(L2) +"," + str(R)+","+str(P3) + "," + str(C3)+ "," + str(L3) +"," + str(C)
        microgear.chat("detail",food1)
        food2 = food_c+","+food_l+","+food_r+","+str(round(Loadcell_w[1],2))+","+str(Loadcell_w[2])+","+str(round(Loadcell_w[3],2))+","+str(Loadcell_w[0])+","+str(round(neuron_w[1],2))+","+str(neuron_w[2])+","+str(neuron_w[3])+","+str(round(neuron_w[0],2))
        microgear.chat("namefood",food2)
        total = 572.5 - ((P3*4)+(C3*4)+(L3*4))
        food3 = str(P3*4)+"kcal"+","+str(C3*4)+"kcal"+","+str(L3*4)+"kcal"+","+str((P3*4)+(C3*4)+(L3*4))+"kcal"+","+str(total)+"kcal"
        microgear.chat("total",food3)
        time.sleep(10)
        microgear.on_disconnect = disconnect
        microgear.disconnect()
   
    time.sleep(1)