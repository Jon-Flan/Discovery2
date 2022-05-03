# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

"""
    File 4
    
    This splits the labeled data in training and test data sets.
    To try keep the balance of true versus negative exolpanets the 
    same in each of the data sets (roughly a 75/25 split)
    sklearn stratified kfold is used to determain the split and then 
    a test ration of 80 train 20 test is created and exported to csv files
    
"""

# importing the data from pre made labeled csv
def import_data():

    file = (r'D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/labeled_data.csv')
    data = pd.read_csv(file, low_memory=False)
    
    return data

# spliting the data 75/25 while perserving the ratio of True/False Labels 
def split_train_test(data: np.ndarray, distribution: list, test_ratio: float):
    
    skf = StratifiedKFold(n_splits=int(test_ratio * 100), random_state=1374, shuffle=True)
    
    return next(skf.split(data, distribution))



# perform the split and return train / test sets
def split(data):
    
    test_ratio = 0.04
    labels = data['Label']
    
    
    split = split_train_test(data, labels, test_ratio)
    
    train = split[0]
    rest = split[1]
    
    df_train = data.iloc[train]
    rest_data = data.iloc[rest]
    
    val_ratio = 0.02
    val_labels = rest_data['Label']
    
    split_val = split_train_test(rest_data, val_labels, val_ratio)
    
    test = split_val[0]
    val = split_val[1]
    
    df_test = rest_data.iloc[test]
    df_val = rest_data.iloc[val]
    
    return df_train, df_test, df_val


def check_pcts(df_val, df_test, df_train, data):    
    
    # validation data percentage split between true and false
    df_val_count_true = df_val['Label'].value_counts()[1]
    df_val_count_false = df_val['Label'].value_counts()[0]
    
    # get pcts
    df_val_pct_true = df_val_count_true / len(df_val)
    df_val_pct_false = df_val_count_false / len(df_val)
    
    # test data percentage split between true and false
    df_test_count_true = df_test['Label'].value_counts()[1]
    df_test_count_false = df_test['Label'].value_counts()[0]
    
    # get pcts
    df_test_pct_true = df_test_count_true / len(df_test)
    df_test_pct_false = df_test_count_false / len(df_test)
    
    
    # train data percentage split between true and false
    df_train_count_true = df_train['Label'].value_counts()[1]
    df_train_count_false = df_train['Label'].value_counts()[0]
    
    # get pcts
    df_train_pct_true = df_train_count_true / len(df_train)
    df_train_pct_false = df_train_count_false / len(df_train)
    
    # Origingal data percent split
    data_count_true = data['Label'].value_counts()[1]
    data_count_false = data['Label'].value_counts()[0]
    
    # get pcts
    data_pct_true = data_count_true / len(data)
    data_pct_false = data_count_false / len(data)
    
    data = {'Set':['Train True', 'Train False',  
                   'Test True', 'Test False',
                   'Val True', 'Val False',
                   'Original True', 'Original False'], 
            'Pct':[df_train_pct_true, df_train_pct_false, 
                   df_test_pct_true, df_test_pct_false, 
                   df_val_pct_true, df_val_pct_false,
                   data_pct_true, data_pct_false]}
    
    df_pcts = pd.DataFrame(data)
    
    return df_pcts

# split the data into test and train
data = import_data().copy()
split_data = split(data)
df_train = split_data[0].copy() 
df_test = split_data[1].copy()
df_val = split_data[2].copy()


# check split percentages
df_pcts = check_pcts(df_val, df_test, df_train, data)

# export the data
df_train.to_csv(r'D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/train_data.csv', index=False)
df_test.to_csv(r'D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/test_data.csv', index=False)
df_val.to_csv(r'D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/val_data.csv', index=False)



