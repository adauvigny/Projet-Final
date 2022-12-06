#Useful functions for the API 

import os

from flask import Flask, request, jsonify
import random
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier


#DATA_FILEPATH = os.path.join('API', 'data', 'cs-training.csv')
MODEL_FILEPATH_A = os.path.join('API', 'models', 'model.joblib')
MODEL_FILEPATH_B = os.path.join('API', 'models', 'model.joblib')

#features engineering 

a ={"id":1, "ball_pos_x": -56.27}


def data_treatment(params):
    #params has to be a dictionary with the 55 features - well ordered
    #{"id":1, "ball_pos_x": -56.27, ...}
    print('nothing')