import json
import numpy as np
import os
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from azureml.core import Model

def init():
    global model
    model_name = 'farepredictor'
    path = Model.get_model_path(model_name)
    model = joblib.load(path)

def run(data):
    try:
        data = json.loads(data)
        result = model.predict(data['data'])
        return {'data' : result.tolist() , 'message' : "Successfully Predicted"}

    except Exception as e:
        error = str(e)
        return {'data' : error , 'message' : 'Failed to Predict'}