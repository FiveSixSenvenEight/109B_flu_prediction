import os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, SimpleRNN, Embedding, Dense, TimeDistributed, GRU, \
                                    Dropout, Bidirectional, Conv1D, BatchNormalization

print(tf.keras.__version__)
print(tf.__version__)


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
            
# Load in the entire flu data for all states
flu_df = pd.read_csv('../data/clean_flu_data.csv')

# Helper functions
def minmax_transform(X_train, X_test):
    """ Uses MinMaxScaler to scale X_train and X_test
        to 0 and 1, scaled using X_train
    """
    scalar = MinMaxScaler().fit(X_train)
    X_test = scalar.transform(X_test)
    X_train = scalar.transform(X_train)
    return X_train, X_test


def get_X(option, state, df_train, df_test, predictor_state_list):
    """ Get data according to option
    """ 
    assert option in ['TS', 'GT', 'other_states']

    # Modify flu_df
    flu_df_train = pd.merge(flu_df.drop(state,1), df_train[['date']], on = 'date', how = 'right')
    flu_df_test = pd.merge(flu_df.drop(state,1), df_test[['date']], on = 'date', how = 'right')
    flu_df_train = flu_df_train.drop('date', 1)
    flu_df_test = flu_df_test.drop('date', 1)

    # Drop all target columns and date column
    target_cols = [c for c in df_train.columns if c.startswith('target')]
    drop_cols = target_cols + ['date']
    df_train = df_train.drop(drop_cols,1)
    df_test = df_test.drop(drop_cols,1)

    # Flu for the current state
    flu_colname = state.lower().replace(" ","_")+"_flu"

    if option == 'TS':
        X_train_ts = df_train[[flu_colname]]
        X_test_ts = df_test[[flu_colname]]
        X_train_ts, X_test_ts = minmax_transform(X_train_ts, X_test_ts)
        X_all_ts = np.concatenate((X_train_ts, X_test_ts))
        return X_train_ts, X_test_ts, X_all_ts
    
    if option == 'GT':
        gt_columns = list(df_train.columns.difference([flu_colname]))
        X_train_gt = df_train[gt_columns]
        X_test_gt = df_test[gt_columns]
        X_train_gt, X_test_gt = minmax_transform(X_train_gt, X_test_gt)
        X_all_gt = np.vstack((X_train_gt, X_test_gt))
        return X_train_gt, X_test_gt, X_all_gt
    
    if option == 'other_states':
        # Default predictor_state_list to be all states other than input state
        if predictor_state_list == None:
            predictor_state_list = flu_df_train.columns
        X_train_states = flu_df_train[predictor_state_list]
        X_test_states = flu_df_test[predictor_state_list]
        X_train_states, X_test_states = minmax_transform(X_train_states, X_test_states)
        X_all_state = np.vstack((X_train_states, X_test_states))
        return X_train_states, X_test_states, X_all_state


def get_y(df_train, df_test, target_lag):
    """Get the target with the desired target_lag
    """
    assert target_lag in [1,2,4,8]
    # Flu for the current state
    target_train = df_train[[f'target_{target_lag}']].values
    target_test = df_test[[f'target_{target_lag}']].values
    target_train, target_test = minmax_transform(target_train, target_test)
    target_all = np.concatenate((target_train, target_test))
    return target_train, target_test, target_all


def get_data(  state,
               option_list = ['TS'],
               target_lag = 1,
               predictor_state_list = None
               ):
    """ Create desired dataset according to option_list for state
        Inputs:
            state: state name 
            option_list: one of ['TS', 'GT', 'other_states'], corresponding to
                time series, google trends, other states flu data
            target_lag: target lag, one of 1,2,4,8
            predictor_state_list: list of other states to consider


    """
    # Get the appropriate df_train and df_test given state
    df_train = train_dfs[state]
    df_test = test_dfs[state]

    # Get all data according to option_list
    X_train, X_test, X_all = get_X(option_list[0], state, df_train, df_test, predictor_state_list)

    for option in option_list[1:]:
        X_train_t, X_test_t, X_all_t = get_X(option, state, df_train, df_test, predictor_state_list)
        X_train = np.hstack((X_train,X_train_t))
        X_test = np.hstack((X_test,X_test_t))
        X_all = np.hstack((X_all,X_all_t))

    # Get the target
    y_train, y_test, y_all = get_y(df_train, df_test, target_lag)

    # Return X, y
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(X_all), pd.DataFrame(y_train), pd.DataFrame(y_test), pd.DataFrame(y_all)









