#Useful functions for the API 

import os
import flask
from flask import Flask, request, jsonify
import random
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from numpy import nan 

#DATA_FILEPATH = os.path.join('API', 'data', 'cs-training.csv')
MODEL_FILEPATH_A = os.path.join('API','app', 'models', 'predictA.joblib')
MODEL_FILEPATH_B = os.path.join('API','app', 'models', 'predictB.joblib')

#features engineering 


a = [{'id': 0, 'ball_pos_x': -56.2708, 'ball_pos_y': 29.51, 'ball_pos_z': 17.3486, 'ball_vel_x': 24.4994, 'ball_vel_y': -1.3113999, 'ball_vel_z': 11.006801, 'p0_pos_x': -35.7762, 'p0_pos_y': 73.1368, 'p0_pos_z': 1.248, 'p0_vel_x': 18.387, 'p0_vel_y': -5.135, 'p0_vel_z': -21.4028, 'p0_boost': 0.0, 'p1_pos_x': -72.9054, 'p1_pos_y': 28.819399, 'p1_pos_z': 11.7, 'p1_vel_x': -19.212801, 'p1_vel_y': -1.8103999, 'p1_vel_z': 5.7040005, 'p1_boost': 49.4, 'p2_pos_x': -36.3792, 'p2_pos_y': -18.8584, 'p2_pos_z': 0.3402, 'p2_vel_x': -45.022797, 'p2_vel_y': -5.5773997, 'p2_vel_z': 0.0042, 'p2_boost': 87.8, 'p3_pos_x': -3.4352, 'p3_pos_y': 93.9754, 'p3_pos_z': 0.3402, 'p3_vel_x': -27.2624, 'p3_vel_y': -2.5516, 'p3_vel_z': 0.005, 'p3_boost': 69.06, 'p4_pos_x': -23.3904, 'p4_pos_y': 101.7156, 'p4_pos_z': 28.9726, 'p4_vel_x': 25.478, 'p4_vel_y': 11.5176, 'p4_vel_z': -18.315401, 'p4_boost': 83.1, 'p5_pos_x': -51.0556, 'p5_pos_y': 54.5942, 'p5_pos_z': 0.34, 'p5_vel_x': 3.9484, 'p5_vel_y': -16.7108, 'p5_vel_z': 0.0074, 'p5_boost': 71.0, 'boost0_timer': 0.0, 'boost1_timer': -3.264, 'boost2_timer': -6.133, 'boost3_timer': -6.875, 'boost4_timer': -7.016, 'boost5_timer': -3.23},{'id': 1.0, 'ball_pos_x': 2.8528, 'ball_pos_y': 70.196, 'ball_pos_z': 8.949, 'ball_vel_x': -8.1522, 'ball_vel_y': -65.5772, 'ball_vel_z': 18.5364, 'p0_pos_x': 22.926, 'p0_pos_y': 87.5438, 'p0_pos_z': 0.3396, 'p0_vel_x': -41.9548, 'p0_vel_y': -18.795, 'p0_vel_z': 0.0114, 'p0_boost': 0.784, 'p1_pos_x': 5.9602003, 'p1_pos_y': 59.6002, 'p1_pos_z': 0.34, 'p1_vel_x': -44.1434, 'p1_vel_y': -12.936601, 'p1_vel_z': 0.0023999999, 'p1_boost': 34.5, 'p2_pos_x': 69.7366, 'p2_pos_y': -11.2536, 'p2_pos_z': 0.3402, 'p2_vel_x': -0.53400004, 'p2_vel_y': -45.6948, 'p2_vel_z': 0.0042, 'p2_boost': 0.0, 'p3_pos_x': nan, 'p3_pos_y': nan, 'p3_pos_z': nan, 'p3_vel_x': nan, 'p3_vel_y': nan, 'p3_vel_z': nan, 'p3_boost': nan, 'p4_pos_x': 12.2516, 'p4_pos_y': 86.967804, 'p4_pos_z': 1.5382, 'p4_vel_x': 27.584, 'p4_vel_y': -26.174599, 'p4_vel_z': -0.2928, 'p4_boost': 84.2, 'p5_pos_x': 39.1266, 'p5_pos_y': 92.815, 'p5_pos_z': 0.3402, 'p5_vel_x': -15.4968, 'p5_vel_y': -14.8766, 'p5_vel_z': 0.005, 'p5_boost': 66.7, 'boost0_timer': 0.0, 'boost1_timer': -1.615, 'boost2_timer': -5.97, 'boost3_timer': -5.504, 'boost4_timer': 0.0, 'boost5_timer': -6.51}]



def predict_scores(input_array): 

    input_data = pd.DataFrame(input_array)
    test_data = input_data.copy()
    test_data['delta_p0_ball'] = np.sqrt((test_data['p0_pos_x'] - test_data['ball_pos_x'])**2 + (test_data['p0_pos_y'] - test_data['ball_pos_y'])**2 + (test_data['p0_pos_z'] - test_data['ball_pos_z'])**2)
    test_data['delta_p1_ball'] = np.sqrt((test_data['p1_pos_x'] - test_data['ball_pos_x'])**2 + (test_data['p1_pos_y'] - test_data['ball_pos_y'])**2 + (test_data['p1_pos_z'] - test_data['ball_pos_z'])**2)
    test_data['delta_p2_ball'] = np.sqrt((test_data['p2_pos_x'] - test_data['ball_pos_x'])**2 + (test_data['p2_pos_y'] - test_data['ball_pos_y'])**2 + (test_data['p2_pos_z'] - test_data['ball_pos_z'])**2)
    test_data['delta_p3_ball'] = np.sqrt((test_data['p3_pos_x'] - test_data['ball_pos_x'])**2 + (test_data['p3_pos_y'] - test_data['ball_pos_y'])**2 + (test_data['p3_pos_z'] - test_data['ball_pos_z'])**2)
    test_data['delta_p4_ball'] = np.sqrt((test_data['p4_pos_x'] - test_data['ball_pos_x'])**2 + (test_data['p4_pos_y'] - test_data['ball_pos_y'])**2 + (test_data['p4_pos_z'] - test_data['ball_pos_z'])**2)
    test_data['delta_p5_ball'] = np.sqrt((test_data['p5_pos_x'] - test_data['ball_pos_x'])**2 + (test_data['p5_pos_y'] - test_data['ball_pos_y'])**2 + (test_data['p5_pos_z'] - test_data['ball_pos_z'])**2)
    test_data.drop('id',axis=1,inplace=True)

    test_A = test_data.drop(['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer',
       'boost4_timer', 'boost5_timer','p3_pos_x', 'p3_pos_y', 'p3_pos_z', 'p3_vel_x',
       'p3_vel_y', 'p3_vel_z', 'p3_boost', 'p4_pos_x', 'p4_pos_y', 'p4_pos_z',
       'p4_vel_x', 'p4_vel_y', 'p4_vel_z', 'p4_boost', 'p5_pos_x', 'p5_pos_y',
       'p5_pos_z', 'p5_vel_x', 'p5_vel_y', 'p5_vel_z', 'p5_boost'], axis = 1)

    test_B = test_data.drop(['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer',
       'boost4_timer', 'boost5_timer','p0_pos_x', 'p0_pos_y', 'p0_pos_z', 'p0_vel_x',
       'p0_vel_y', 'p0_vel_z', 'p0_boost', 'p1_pos_x', 'p1_pos_y', 'p1_pos_z',
       'p1_vel_x', 'p1_vel_y', 'p1_vel_z', 'p1_boost', 'p2_pos_x', 'p2_pos_y',
       'p2_pos_z', 'p2_vel_x', 'p2_vel_y', 'p2_vel_z', 'p2_boost'], axis = 1)

    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    test_A[test_A.columns] = imp.fit_transform(test_A[test_A.columns])
    test_B[test_B.columns] = imp.fit_transform(test_B[test_B.columns])

    modelA = load(MODEL_FILEPATH_A)
    modelB = load(MODEL_FILEPATH_B)

    A_pred = modelA.predict(test_A)
    B_pred = modelB.predict(test_B)


    d = {'id': input_data.loc[:,'id'].values.tolist(), 'team_A_scoring_within_10sec': A_pred, 'team_B_scoring_within_10sec': B_pred}
    submission = pd.DataFrame(d)
    formatted_submission = submission.to_json(orient="records")
    print(formatted_submission)
    return formatted_submission


predict_scores(a)








# def predict_scores(input_array):
#     scores = []
#     for i in range(len(input_array)):
#         dictionary = input_array[i]
#         prepared_data = data_treatment(dictionary)
#         score = predict(prepared_data)
#         scores.append({'id':dictionary['id'], "team_A_scoring_within_10sec": score[0], "team_B_scoring_within_10sec": score[1]})
#     print(scores)
#     return scores


# def data_treatment(input_data):
#     #params has to be a dictionary with the 55 features - well ordered
#     #{"id":1, "ball_pos_x": -56.27, ...}
#     input_data.pop('id')

#     input_data['delta_p0_ball'] = np.sqrt((input_data['p0_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p0_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p0_pos_z'] - input_data['ball_pos_z'])**2)
#     input_data['delta_p1_ball'] = np.sqrt((input_data['p1_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p1_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p1_pos_z'] - input_data['ball_pos_z'])**2)
#     input_data['delta_p2_ball'] = np.sqrt((input_data['p2_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p2_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p2_pos_z'] - input_data['ball_pos_z'])**2)
#     input_data['delta_p3_ball'] = np.sqrt((input_data['p3_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p3_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p3_pos_z'] - input_data['ball_pos_z'])**2)
#     input_data['delta_p4_ball'] = np.sqrt((input_data['p4_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p4_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p4_pos_z'] - input_data['ball_pos_z'])**2)
#     input_data['delta_p5_ball'] = np.sqrt((input_data['p5_pos_x'] - input_data['ball_pos_x'])**2 + (input_data['p5_pos_y'] - input_data['ball_pos_y'])**2 + (input_data['p5_pos_z'] - input_data['ball_pos_z'])**2)

#     keys_to_remove_A = ['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer',
#        'boost4_timer', 'boost5_timer','p3_pos_x', 'p3_pos_y', 'p3_pos_z', 'p3_vel_x',
#        'p3_vel_y', 'p3_vel_z', 'p3_boost', 'p4_pos_x', 'p4_pos_y', 'p4_pos_z',
#        'p4_vel_x', 'p4_vel_y', 'p4_vel_z', 'p4_boost', 'p5_pos_x', 'p5_pos_y',
#        'p5_pos_z', 'p5_vel_x', 'p5_vel_y', 'p5_vel_z', 'p5_boost']

#     keys_to_remove_B = ['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer',
#        'boost4_timer', 'boost5_timer','p0_pos_x', 'p0_pos_y', 'p0_pos_z', 'p0_vel_x',
#        'p0_vel_y', 'p0_vel_z', 'p0_boost', 'p1_pos_x', 'p1_pos_y', 'p1_pos_z',
#        'p1_vel_x', 'p1_vel_y', 'p1_vel_z', 'p1_boost', 'p2_pos_x', 'p2_pos_y',
#        'p2_pos_z', 'p2_vel_x', 'p2_vel_y', 'p2_vel_z', 'p2_boost']

#     prepared_data_A = input_data.copy()
#     for key in keys_to_remove_A: 
#         prepared_data_A.pop(key)

#     prepared_data_B = input_data.copy() 
#     for key in keys_to_remove_B: 
#         prepared_data_B.pop(key)

#     imp = SimpleImputer(missing_values=np.nan, strategy='median')
#     prepared_data_A[prepared_data_A.columns] = imp.fit_transform(prepared_data_A[prepared_data_A.columns])
#     prepared_data_B[prepared_data_B.columns] = imp.fit_transform(prepared_data_B[prepared_data_B.columns])


#     print('nothing')

#     return(prepared_data_A, prepared_data_B)

# def predict(prepared_data):
#     prepared_data_A = prepared_data[0]
#     prepared_data_B = prepared_data[1]

#     modelA = load(MODEL_FILEPATH_A)
#     modelB = load(MODEL_FILEPATH_B)
#     prediction_A = modelA.predict_proba(prepared_data_A)
#     prediction_B = modelB.predict_proba(prepared_data_B)

#     return(prediction_A, prediction_B)

