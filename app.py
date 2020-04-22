from flask import Flask
from flask import render_template, request, jsonify, url_for
from keras.models import load_model
import flask
import numpy as np
import traceback
import pickle
import pandas as pd

app = Flask(__name__,template_folder='templates')

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]

#init
model_columns = ["left_sensor","top_sensor","right_sensor"]
model_predicts = load_model('models/model_weight.h5')

#read_file normalized and invest value
max_data = []
file_opject =  open("models/model_weight_max.txt","r")
num_max = file_opject.read()
file_opject.close()

list_data_max =  num_max.split(",")
i = 0 
while(i < len(list_data_max) ):
    max_data.append(float(list_data_max[i+1]))
    i = i+2

#normalize for predict
normal = pd.DataFrame(max_data[:3])
#invest-normalize from predict
invest = pd.DataFrame(max_data[3:])


 
 
@app.route('/')
@app.route('/home')
def home():
    return render_template('main.html' , posts = posts) 

@app.route('/about')
def about():
    return render_template('about.html' , title = 'About')

@app.route('/predict', methods=['POST','GET'])
def predict():
  
   if flask.request.method == 'GET':
       return "Prediction page"
 
   if flask.request.method == 'POST':
       try:
           json_ = request.json
           print(json_)
           query_ = pd.get_dummies(pd.DataFrame(json_))
        #    print(query_)
           query = query_.reindex(columns = model_columns, fill_value= 0)
           print(query[0])
           prediction = list(model_predicts.predict(query))
 
           return jsonify({
               "prediction":str(prediction)
           })
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
      
 
if __name__ == "__main__":
   app.run()