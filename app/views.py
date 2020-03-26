from app import app
from flask import render_template, request, jsonify
import flask
import numpy as np
import traceback
from keras.models import load_model
import pandas as pd
import json

# import firebase_admin
# from firebase_admin import credentials

# cred = credentials.Certificate("path/to/serviceAccountKey.json")
# firebase_admin.initialize_app(cred)

@app.route('/')
@app.route('/index')
@app.route('/predict', methods=['POST','GET'])



def index():
    return "Hello, World!"


def predict():
   gg = ""
   if flask.request.method == 'GET':
       return "Prediction page"
 
   if flask.request.method == 'POST':
       try:
           namefile = "model_weight"
        #    path_file = "food"
        #    weight = ["left_sensor","top_sensor","right_sensor","left_total","top_total","right_total"]
           mlp = load_model("models/"+namefile+".h5")
           json_ = request.json
           gg = json_
           json_str = json.loads(json_) 
        #    query_ = pd.get_dummies(pd.DataFrame(json_))
        #    print(query_)
           query = [int(json_str["left"]),int(json_str["centor"]),int(json_str["right"])]
        #    query = query_.reindex(columns = model_columns, fill_value= 0)
           prediction = mlp.predict(query)[0]
 
           return jsonify({
               "prediction":str(prediction)
           })
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
            #    "json_": str(gg)
               })
# def home():
#     return ""