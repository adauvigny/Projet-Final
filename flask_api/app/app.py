#the real app

import os

from flask import Flask, request, jsonify
import random
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
import functions
import joblib

#DATA_FILEPATH = os.path.join('API', 'data', 'cs-training.csv')
ROOT_APP_FOLDER = "/opt/app"


MODEL_FILEPATH_A = os.path.join(ROOT_APP_FOLDER,'models', 'predictA.joblib')
MODEL_FILEPATH_B = os.path.join(ROOT_APP_FOLDER,'models', 'predictB.joblib')



app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the app by Theophile & Alex"


@app.route('/predict', methods=['GET'])
def get_score():
    score = random.uniform(0, 1)
    return jsonify({"score": score})


@app.route('/predict', methods=['POST'])
def get_scores():
    payload = request.json
    print(payload)
    scores = functions.predict_scores(payload)
    return scores

