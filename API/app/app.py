#the real app

import os

from flask import Flask, request, jsonify
import random
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier

#DATA_FILEPATH = os.path.join('API', 'data', 'cs-training.csv')
MODEL_FILEPATH_A = os.path.join('API', 'app','models', 'predictA.joblib')
MODEL_FILEPATH_B = os.path.join('API', 'app','models', 'predictB.joblib')



app = Flask(__name__)