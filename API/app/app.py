#the real app

import os

from flask import Flask, request, jsonify
import random
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
import functions
#DATA_FILEPATH = os.path.join('API', 'data', 'cs-training.csv')
MODEL_FILEPATH_A = os.path.join('API', 'app','models', 'predictA.joblib')
MODEL_FILEPATH_B = os.path.join('API', 'app','models', 'predictB.joblib')



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


response = requests.get('http://127.0.0.1:5000/predict')
print(response.text)

