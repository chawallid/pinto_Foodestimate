import sys
sys.path.insert(0,"c:/python37/lib/site-packages/")
import logging
import time
import random
import microgear.client as microgear
import pyrebase

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
store.child("food/left.jpe").put("img/left.jpg")
store.child("food/rigth.jpe").put("img/right.jpg")
store.child("food/center.jpe").put("img/center.jpg")

url_L = store.child("food/left.jpe").get_url(None)
print(url_L)
url_R = store.child("food/rigth.jpe").get_url(None)
url_C = store.child("food/center.jpe").get_url(None)
print("INFO : Upload Success !!!")
# db.child("food").child("left").update({"url":"url"})
# db.child("food").child("right").update({"url":"url"})
# db.child("food").child("center").update({"url":"url"})

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
# list_data = ["a","b"]
microgear.setalias("a")
# microgear.setalias("b")

microgear.on_connect = connection
microgear.on_message = subscription
microgear.on_disconnect = disconnect
microgear.subscribe("/mails")

microgear.connect()
L = url_L + "&token=28ef2706-806f-4699-afe1-ff737ac9b4da"
R = url_R + "&token=d28ba572-312b-468e-a234-354b96d14319"
C = url_C + "&token=0b0a32a9-1a00-43c0-a665-19058365ff14"
if(L == url_L):
    print("correct")
print(url_L)   
while True:
    P1 = random.randint(0,100)		
    C1 = random.randint(0,100)
    L1 = random.randint(0,100)

    P2 = random.randint(0,100)		
    C2 = random.randint(0,100)
    L2 = random.randint(0,100)

    P3 = random.randint(0,100)		
    C3 = random.randint(0,100)
    L3 = random.randint(0,100)

    food1 = str(P1) + "," + str(C1)+ "," + str(L1) +"," + str(L)+","+str(P2) + "," + str(C2)+ "," + str(L2) +"," + str(R)+","+str(P3) + "," + str(C3)+ "," + str(L3) +"," + str(C)
    microgear.chat("a",food1)
    time.sleep(30)