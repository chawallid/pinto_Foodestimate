import sys
sys.path.insert(0,"c:/python37/lib/site-packages/")
import logging
import time
import random
import microgear.client as microgear
import pyrebase
import main as pinto

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
C,img_C,L,img_L,R,img_R = pinto.main_function()
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
gearkey = "ZlLVTgx1SCXW1xJ"
gearsecret =  "E4gnN6yKtdG0DC8WWDDFnhR5q"
print("INFO : Connecting Netpie !!!")

microgear.create(gearkey,gearsecret,appid,{'debugmode': True})

def connection():
    logging.info("Now I am connected with netpie")

def subscription(topic,message):
    logging.info(topic+" "+message)

def disconnect():
    logging.debug("disconnect is work")

microgear.setalias("a")
microgear.on_connect = connection
microgear.on_message = subscription
microgear.on_disconnect = disconnect
microgear.subscribe("/mails")
microgear.connect(False)

#token_firebase
L = url_L + "&token=28ef2706-806f-4699-afe1-ff737ac9b4da"
R = url_R + "&token=d28ba572-312b-468e-a234-354b96d14319"
C = url_C + "&token=0b0a32a9-1a00-43c0-a665-19058365ff14"

while True:
    food1 = str(P1) + "," + str(C1)+ "," + str(L1) +"," + str(L)+","+str(P2) + "," + str(C2)+ "," + str(L2) +"," + str(R)+","+str(P3) + "," + str(C3)+ "," + str(L3) +"," + str(C)
    microgear.chat("a",food1)
    time.sleep(30)