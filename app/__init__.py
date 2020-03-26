from flask import Flask
# from flask import render_template, request, jsonify
# import flask
# import numpy as np
# import traceback
# from keras.models import load_model
# import pandas as pd

app = Flask(__name__,template_folder='templates')
# mlp = load_model("models/"+namefile+".h5")
from app import views