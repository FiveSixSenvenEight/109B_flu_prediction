import os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


def get_X(option, state, predictor_state_list):
    """ Get data according to option
    """ 
    assert option in ['TS', 'GT', 'other_states'] or option.startswith('lag_')

    # Get dfs
    df_train, df_test = train_dfs[state], test_dfs[state]

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
    if option.startswith('lag_'):
        n_lag = int(option[4:])
        X_train_ts = df_train[[flu_colname]].shift(n_lag).fillna(method='bfill')
        X_test_ts = df_test[[flu_colname]].shift(n_lag).fillna(method='bfill')
        X_train_ts, X_test_ts = minmax_transform(X_train_ts, X_test_ts)
        X_all_ts = np.concatenate((X_train_ts, X_test_ts))
        assert np.isnan(X_all_ts.flatten()).sum() == 0
        return X_train_ts, X_test_ts, X_all_ts

def get_y(state, target_lag):
    """Get the target with the desired target_lag
    """
    assert target_lag in [1,2,4,8]
    # Flu for the current state
    df_train, df_test = train_dfs[state], test_dfs[state]
    target_train = df_train[[f'target_{target_lag}']].values
    target_test = df_test[[f'target_{target_lag}']].values
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
    X_train, X_test, X_all = get_X(option_list[0], state, predictor_state_list)

    for option in option_list[1:]:
        X_train_t, X_test_t, X_all_t = get_X(option, state, predictor_state_list)
        X_train = np.hstack((X_train,X_train_t))
        X_test = np.hstack((X_test,X_test_t))
        X_all = np.hstack((X_all,X_all_t))

    # Get the target
    y_train, y_test, y_all = get_y(state, target_lag)

    # Return X, y
    return X_train, X_test, X_all, y_train, y_test, y_all

def inverse_transform_preds(y_train, y_pred):
    """ Uses MinMaxScaler to scale y_pred to the 
        original range as y_train
    """
    scalar = MinMaxScaler().fit(y_train)
    transformed_preds = scalar.inverse_transform(y_pred)
    return transformed_preds

def plot_predictions(y_train, y_test, model_name, y_pred, model_name_gt = None, y_pred_gt = None):
    title = f"{model_name} Performance"
    print(f'Test RMSE for {model_name}: ', np.sqrt(mean_squared_error(y_test, y_pred)))
    y_train = y_train[-100:]
    train_len = len(y_train)
    test_len = len(y_test)
    
    plt.figure(figsize=(20,8))
    plt.plot(range(train_len), y_train, label = 'y_train')
    plt.plot(range(train_len, train_len + test_len), y_test, c='g', label = 'y_test')
    plt.plot(range(train_len, train_len + test_len), y_pred, c='orange', ls='--', label = 'y_pred')
    if y_pred_gt is not None:
        print(f'Test RMSE for {model_name_gt}: ', np.sqrt(mean_squared_error(y_test, y_pred_gt)))
        plt.plot(range(train_len, train_len + test_len), y_pred_gt, c='red', ls='--', label = 'y_pred gt')
        title = f"{model_name} & {model_name_gt} Performance"
    plt.legend()
    plt.title(title)