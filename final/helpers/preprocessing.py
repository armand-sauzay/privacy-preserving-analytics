import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import linear_model
import joblib
from sklearn import metrics

def create_train_test(df
    , features=[#'Timestamp_bid'
    #, 'User-Agent_bid'
    #'City ID'
    'Ad Exchange'
    , 'Ad Slot Width'
    , 'Ad Slot Height'
    , 'Ad Slot Visibility'
    , 'Ad Slot Format'
    , 'Ad Slot Floor Price'
    , 'Bidding Price'
    , 'Paying Price'
    , 'Advertiser ID'
    #, '10110' 
        ]+['10006','10024','10031','10048','10052','10057','10059','10063','10067','10074','10075','10076','10077','10079','10083','10093','10102','10110','10111','10684','11092','11278','11379','11423','11512','11576','11632','11680','11724','11944','13042','13403','13496','13678','13776','13800','13866','13874','14273','16593','16617','16661','16706']
    , target='has click'):
    features_target=features+[target]

    #full datset 

    #df.fillna(0,inplace=True)
    df_red=df.filter(features_target)
    df_red.dropna(inplace=True)
    n = len(df_red)


    df_train = df_red[0:int(n*0.8)]
    df_test = df_red[int(n*0.8):]

    return(df_train, df_test)

def create_X_y(df_train, df_test, features=[#'Timestamp_bid'
            #, 'User-Agent_bid'
            #'City ID'
            'Ad Exchange'
            , 'Ad Slot Width'
            , 'Ad Slot Height'
            , 'Ad Slot Visibility'
            , 'Ad Slot Format'
            , 'Ad Slot Floor Price'
            , 'Bidding Price'
            , 'Paying Price'
            , 'Advertiser ID'
            #, '10110' 
                ]+['10006','10024','10031','10048','10052','10057','10059','10063','10067','10074','10075','10076','10077','10079','10083','10093','10102','10110','10111','10684','11092','11278','11379','11423','11512','11576','11632','11680','11724','11944','13042','13403','13496','13678','13776','13800','13866','13874','14273','16593','16617','16661','16706']
            , target='has click'):
    features_target=features+[target]
    X_train=df_train.filter(features)
    X_test=df_test.filter(features)

    y_train=df_train[target]
    y_test=df_test[target]

    return(X_train, X_test, y_train, y_test)


def create_small_X_y(df_train, df_test
    , features=[#'Timestamp_bid'
    #, 'User-Agent_bid'
    #'City ID'
    'Ad Exchange'
    , 'Ad Slot Width'
    , 'Ad Slot Height'
    , 'Ad Slot Visibility'
    , 'Ad Slot Format'
    , 'Ad Slot Floor Price'
    , 'Bidding Price'
    , 'Paying Price'
    , 'Advertiser ID'
    #, '10110' 
        ]+['10006','10024','10031','10048','10052','10057','10059','10063','10067','10074','10075','10076','10077','10079','10083','10093','10102','10110','10111','10684','11092','11278','11379','11423','11512','11576','11632','11680','11724','11944','13042','13403','13496','13678','13776','13800','13866','13874','14273','16593','16617','16661','16706']
    , target='has click'):
    features_target=features+[target]

    X_train_v1=df_train.filter(features)[:10000]
    X_test_v1=df_test.filter(features)[:1000]

    y_train_v1=df_train[target][:10000]
    y_test_v1=df_test[target][:1000]
    
    return(X_train_v1, X_test_v1, y_train_v1, y_test_v1)


    