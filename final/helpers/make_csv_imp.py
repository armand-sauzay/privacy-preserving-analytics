
"""
This script aims at creating a training and a testing set from the iPinYou data contest

This script is composed of 4 parts 
1. Import librairies and change directory to where the data lies 
    ACTION REQUIRED: you need to update the path to your folder 

2. Define parameters 

3. Define function for extracting day of the data and sampling it 

4. Script to build full dataset 
ACTION REQUIRED: 
- Update random state in parameters  
- Update n_sampling 


Author: Armand Sauzay
Email: armand.sauzay@berkeley.edu
"""
# 1. Import librairies
import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import bz2
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import json



#3. one day data
def make_one_day_dataset(date, random_state=0, imp_frac_to_take=0.01, n_sampling=100000):
    #Change directory to where the data lies (training2nd --> 2nd session of contest)
    data_shapes={}
    os.chdir('/Users/Armand/Capstone/ipinyou.contest.dataset/training2nd')

    # 2. Define Parameters
    header_season2=[
        'Bid ID'
        ,'Timestamp'
        ,'Log Type'
        ,'iPinYou ID'
        ,'User-Agent'
        ,'IP'
        ,'Region ID'
        ,'City ID'
        ,'Ad Exchange'
        ,'Domain'
        ,'URL'
        ,'Anonymous URL'
        ,'Ad Slot ID'
        ,'Ad Slot Width'
        ,'Ad Slot Height'
        ,'Ad Slot Visibility'
        ,'Ad Slot Format'
        ,'Ad Slot Floor Price'
        ,'Creative ID'
        ,'Bidding Price'
        ,'Paying Price'
        ,'Landing Page URL'
        ,'Advertiser ID'
        ,'User Profile IDs']


    # READ DATA FOR 1 DAY 
    print('reading data for %s'%date)
    #unzipped_file = bz2.BZ2File('bid.'+date+'.txt.bz2', "r")
    #bids=pd.read_table(unzipped_file, names=header_season2_bids)

    unzipped_file = bz2.BZ2File('imp.'+date+'.txt.bz2', "r")
    impressions=pd.read_table(unzipped_file, names=header_season2)
    
    unzipped_file = bz2.BZ2File('clk.'+date+'.txt.bz2', "r")
    clicks=pd.read_table(unzipped_file, names=header_season2)
    
    #unzipped_file = bz2.BZ2File('conv.'+date+'.txt.bz2', "r")
    #conversions=pd.read_table(unzipped_file, names=header_season2)

    all_data={'impressions':impressions, 'clicks':clicks}
    for key, value in all_data.items(): 
        print('shape of %s is %s'%(key, value.shape))

    #ASSUMPTION 1: SOME COLUMNS CAN BE DROPPED BECAUSE THEY HAVE NO PREDICTIVE POWER
    cols_to_drop=['iPinYou ID'
                ,'IP'
                ,'Domain'
                ,'URL'
                ,'Anonymous URL'
                ,'Ad Slot ID'
                ,'Creative ID']

    for index, value in all_data.items(): 
        value.drop(cols_to_drop, axis=1, inplace =True)
        if index!='bids': 
            value.drop('Landing Page URL', axis=1, inplace=True)

# IMPRESSION MERGE 
    ## ASSUMPTION 2.1:COUNT DUPLICATES ON BID ID FOR IMPRESSIONS AND ADD THEM TO LINE
    ## ASSUMPTION 2.2: ONLY KEEP COLUMNS THAT ACTUALLY DIFFER IN THE IMPRESSION TABLE
    
    '''impression_count=impressions.groupby(['Bid ID']).size()
    #impression_to_merge =pd.merge(impressions[['Bid ID'
            , 'Timestamp'
            , 'User-Agent'
            , 'Log Type'
            , 'Paying Price'
            , 'User Profile IDs']].drop_duplicates('Bid ID'), impression_count.rename('n_impressions'), left_on='Bid ID',right_index=True)
    #df=pd.merge(bids, impression_to_merge, on="Bid ID", suffixes=('_bid', '_imp'), how="left")'''

#CLICKS MERGE
    ## ASSUMPTION 3.1: COUNT DUPLICATES FOR BID ID FOR CLICKS AND ADD THEM  
    ## ASSUMPTION 3.2: ONLY KEEP COLUMNS THAT ACTUALLY DIFFER IN THE CLICKS TABLE
    clicks_count=clicks.groupby(['Bid ID']).size()
    clicks_to_merge =pd.merge(clicks[['Bid ID'
                        , 'Timestamp'
                        , 'User-Agent'
                        , 'User Profile IDs'
                        , 'Region ID'
                        ]].drop_duplicates('Bid ID'), clicks_count.rename('n_clicks'), left_on='Bid ID',right_index=True)     
    df=pd.merge(impressions, clicks_to_merge, left_on="Bid ID",right_on="Bid ID", suffixes=('_impressions', '_clicks'), how="left")

    # ADD COLUMNS (QUITE REDUNDANT WITH n_clicks & n_impressions)
    #df['has impression']=~(df['Timestamp_imp'].isna())
    df['has click']=~(df['Region ID_clicks'].isna())
    #CONVERT BOOL TO INT 
    #df['has impression']=df['has impression'].astype(int)
    df['has click']=df['has click'].astype(int)

    #REDUCE SIZE   
    cols_to_drop = ['Bid ID']
    df.drop(cols_to_drop, axis=1, inplace =True)


    #SAMPLE THE DATASET
    #df=df.sample(n=n_sampling, random_state=random_state)
    n_1=df['has click'].sum()
    #print('n_1: %s'%n_1)
    n_0=int(imp_frac_to_take*(len(df)-n_1))
    #print('n_0: %s'%n_0)
    X=df.drop(['has click'], axis=1)
    y=df['has click']
    rus = RandomUnderSampler(sampling_strategy={1:n_1, 0:n_0}, random_state=0)
    X_res, y_res = rus.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    df=pd.concat([X_res,y_res], axis=1)

    #LOG data shapes 
    data_shapes[date]={}
    data_shapes[date]['impression']=len(impressions)
    data_shapes[date]['impression sampled']=len(df)-df['has click'].sum()

    data_shapes[date]['clicks']=len(clicks)
    data_shapes[date]['clicks sampled']=df['n_clicks'].sum()

    data_shapes[date]['click?']=clicks['Bid ID'].nunique()
    data_shapes[date]['click? sampled']=df['has click'].sum()

    return(data_shapes, df)



def make_full_week_dataset(
    dates_list=['20130606'
            , '20130607'
            , '20130608'
            , '20130609'
            , '20130610'
            , '20130611'
            , '20130612']
    , imp_frac_to_take=0.01):
    datashapes_final, df_final=make_one_day_dataset(dates_list[0], imp_frac_to_take=imp_frac_to_take)
    for date in dates_list[1:]:    
        datashapes, df = make_one_day_dataset(date, imp_frac_to_take=imp_frac_to_take)
        print('shape of df is %s'%str(df.shape))
        datashapes_final[date]=datashapes[date]
        df_final=df_final.append(df)
        print('shape of df_final is %s'%str(df_final.shape))

        #take mapping for user profile
    #mapping=pd.read_table('/Users/Armand/Capstone/ipinyou.contest.dataset/user.profile.tags.en.txt', names=('key','value'))
    df_final.reset_index(drop=True, inplace=True)
    df_final.to_csv('/Users/Armand/Capstone/privacy-preserving-analytics/final/0.dataset/raw/impression_balanced_'+str(imp_frac_to_take)+'_no_dummies.csv')
    np.save('/Users/Armand/Capstone/privacy-preserving-analytics/final/0.dataset/logs/impression_balanced_'+str(imp_frac_to_take)+'_no_dummies_log.npy', datashapes_final)
    
        #dummify user profile
    sorted_df=df_final.sort_values(by='Timestamp_impressions', ascending=True)
    sorted_df=sorted_df[sorted_df['User Profile IDs_impressions'].isna()==0]
    s=sorted_df['User Profile IDs_impressions'].str.split(pat=',')
    dummies_to_concatenate=pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df_dummy=pd.concat([sorted_df, dummies_to_concatenate], axis=1)
    df_dummy.to_csv('/Users/Armand/Capstone/privacy-preserving-analytics/final/0.dataset/dummy/impression_balanced_'+str(imp_frac_to_take)+'.csv')

    return(datashapes_final, df_final)

def dummify(df, imp_frac_to_take):
    sorted_df=df.sort_values(by='Timestamp_impressions', ascending=True)
    sorted_df=sorted_df[sorted_df['User Profile IDs_impressions'].isna()==0]
    s=sorted_df['User Profile IDs_impressions'].str.split(pat=',')
    dummies_to_concatenate=pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df_dummy=pd.concat([sorted_df, dummies_to_concatenate], axis=1)
    df_dummy.to_csv('/Users/Armand/Capstone/privacy-preserving-analytics/final/0.dataset/dummy/impression_balanced_'+str(imp_frac_to_take)+'.csv')

if __name__=="__main__":
    make_full_week_dataset()