import os
from ast import literal_eval
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from data_creater import *
from sklearn.preprocessing import MinMaxScaler

# Load in dfs for all states
all_states = [] # List of all the states
train_dfs, test_dfs = {}, {}
train_path = '../data/state_flu_google/train/'
test_path = '../data/state_flu_google/test/'

for root,dirs,files in os.walk(train_path):
    for file in files:
        if file.endswith('csv'):
            state = file[:-4]
            all_states.append(state)
            train_dfs[state] = pd.read_csv(train_path + file)

for root,dirs,files in os.walk(test_path):
    for file in files:
        if file.endswith('csv'):
            state = file[:-4]
            test_dfs[state] = pd.read_csv(test_path + file)

# Helper functions-normalize the data
def minmax_transform(X_train, X_test, return_scaler=False):
    """ Uses MinMaxScaler to scale X_train and X_test
        to 0 and 1, scaled using X_train
    """
    col_name = X_train.columns
    scaler = MinMaxScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)
    if return_scaler:
        return pd.DataFrame(X_train), pd.DataFrame(X_test), scaler
    else:
        return pd.DataFrame(X_train), pd.DataFrame(X_test)


def get_data(state, future_week, with_gt=False, predictor_state=None):
    '''
    state: state to predict
    future_week: the future week to forcast, chosen from [1,2,4,8]
    with_gt: whether to include GT data
    predictor_state: list of states as predictors
    '''
    assert future_week in [1,2,4,8]
    
    train, test = train_dfs[state], test_dfs[state]
    X_train, X_test = pd.DataFrame(), pd.DataFrame()
    
    # if not specified, only include the flu data as predictor
    flu_train, flu_test, scaler = minmax_transform(pd.DataFrame(train.iloc[:, 1]), pd.DataFrame(test.iloc[:, 1]),
                                                  return_scaler=True)
    X_train['flu_data'], X_test['flu_data'] = flu_train[0], flu_test[0]
    target = 'target_' + str(future_week)
    y_train, y_test = pd.DataFrame(scaler.transform(train[[target]])), test[[target]]
    
    if with_gt and predictor_state:
        for p in predictor_state:
            # flu
            state_train, state_test = train_dfs[p], test_dfs[p]
            flu_train, flu_test = minmax_transform(
                pd.DataFrame(state_train.iloc[:, 1]), pd.DataFrame(state_test.iloc[:, 1]))
            X_train[p], X_test[p] = flu_train, flu_test
            # google trend
            X_train_gt, X_test_gt = state_train.iloc[:, 2:-4], state_test.iloc[:, 2:-4] 
            gt_train, gt_test = minmax_transform(X_train_gt, X_test_gt)
            for i, col in enumerate(X_train_gt.columns):
                X_train[p+' '+col], X_test[p+' '+col] = gt_train[i], gt_test[i]
    
    elif with_gt: # include google trend data
        X_train_gt, X_test_gt = train.iloc[:, 2:-4], test.iloc[:, 2:-4] # google trend data
        gt_train, gt_test = minmax_transform(X_train_gt, X_test_gt)
        for i, col in enumerate(X_train_gt.columns):
            X_train[col], X_test[col] = gt_train[i], gt_test[i]
            
    elif predictor_state:
        for p in predictor_state:
            state_train, state_test = minmax_transform(
                pd.DataFrame(train_dfs[p].iloc[:, 1]), pd.DataFrame(test_dfs[p].iloc[:, 1]))
            X_train[p], X_test[p] = state_train, state_test
            
    X_all, y_all = pd.concat([X_train, X_test], ignore_index=True), pd.concat([y_train.iloc[:, 0], y_test.iloc[:, 0]], ignore_index=True)
    
    return X_train, X_test, X_all, y_train, y_test, y_all, scaler