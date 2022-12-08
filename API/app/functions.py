#Useful functions for the API 

import os

from flask import Flask, request, jsonify
import random
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


#DATA_FILEPATH = os.path.join('API', 'data', 'cs-training.csv')
MODEL_FILEPATH_A = os.path.join('API','app', 'models', 'predictA.joblib')
MODEL_FILEPATH_B = os.path.join('API','app', 'models', 'predictB.joblib')

#features engineering 

a ={"id":1, "ball_pos_x": -56.27}


def data_treatment(input_data):
    #params has to be a dictionary with the 55 features - well ordered
    #{"id":1, "ball_pos_x": -56.27, ...}
    input_data.pop('id')

    input_data['delta_p0_ball'] = np.sqrt((input_data['p0_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p0_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p0_pos_z'] - input_data['ball_pos_z'])**2)
    input_data['delta_p1_ball'] = np.sqrt((input_data['p1_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p1_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p1_pos_z'] - input_data['ball_pos_z'])**2)
    input_data['delta_p2_ball'] = np.sqrt((input_data['p2_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p2_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p2_pos_z'] - input_data['ball_pos_z'])**2)
    input_data['delta_p3_ball'] = np.sqrt((input_data['p3_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p3_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p3_pos_z'] - input_data['ball_pos_z'])**2)
    input_data['delta_p4_ball'] = np.sqrt((input_data['p4_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p4_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p4_pos_z'] - input_data['ball_pos_z'])**2)
    input_data['delta_p5_ball'] = np.sqrt((input_data['p5_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p5_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p5_pos_z'] - input_data['ball_pos_z'])**2)

    keys_to_remove_A = ['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer',
       'boost4_timer', 'boost5_timer','p3_pos_x', 'p3_pos_y', 'p3_pos_z', 'p3_vel_x',
       'p3_vel_y', 'p3_vel_z', 'p3_boost', 'p4_pos_x', 'p4_pos_y', 'p4_pos_z',
       'p4_vel_x', 'p4_vel_y', 'p4_vel_z', 'p4_boost', 'p5_pos_x', 'p5_pos_y',
       'p5_pos_z', 'p5_vel_x', 'p5_vel_y', 'p5_vel_z', 'p5_boost']

    keys_to_remove_B = ['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer',
       'boost4_timer', 'boost5_timer','p0_pos_x', 'p0_pos_y', 'p0_pos_z', 'p0_vel_x',
       'p0_vel_y', 'p0_vel_z', 'p0_boost', 'p1_pos_x', 'p1_pos_y', 'p1_pos_z',
       'p1_vel_x', 'p1_vel_y', 'p1_vel_z', 'p1_boost', 'p2_pos_x', 'p2_pos_y',
       'p2_pos_z', 'p2_vel_x', 'p2_vel_y', 'p2_vel_z', 'p2_boost']

    prepared_data_A = input_data 
    for key in keys_to_remove_A: 
        prepared_data_A.pop(key)

    prepared_data_B = input_data 
    for key in keys_to_remove_B: 
        prepared_data_B.pop(key)

    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    prepared_data_A[prepared_data_A.columns] = imp.fit_transform(prepared_data_A[prepared_data_A.columns])
    prepared_data_B[prepared_data_B.columns] = imp.fit_transform(prepared_data_B[prepared_data_B.columns])


    print('nothing')

    return(prepared_data_A, prepared_data_B)

def predict(prepared_data):
    prepared_data_A = prepared_data[0]
    prepared_data_B = prepared_data[1]

    modelA = load(MODEL_FILEPATH_A)
    modelB = load(MODEL_FILEPATH_B)
    prediction_A = modelA.predict_proba(prepared_data_A)
    prediction_B = modelB.predict_proba(prepared_data_B)

    return(prediction_A, prediction_B)



