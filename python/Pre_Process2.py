# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""

import pandas as pd
import numpy as np

"""
    File 3
    
    This file applies the labeling to the data produced in the pre processing and labels the 
    confirmed planets as 1 and non planets as 0 by using a combination of the disposition and 
    p disposition columns, 
    
    The output as a dataframe with 2001 columns 1 column for the label and 2000 normalised flux readings
    for machine learning input.
    
"""

# method to inport the data
def import_data():

    file = ('D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/candidates_with_flux.csv')
    df = pd.read_csv(file, low_memory=False)
    
    return df


# reorder all the basic info columns to the left of the dataframe
def re_order(frame):
    # move these columns to the start or the dataframe for ease of reading later
    col_list = ['ID','kepoi_name','disposition','p_disposition']
    
    # for each column in the list pop it off and add it to the begining
    for x in range(len(col_list)):
        col_to_move = frame.pop(col_list[x])
        frame.insert(x, col_list[x], col_to_move)
    
    return frame


def relabel_data():
    # import dataframe
    df = import_data().copy()
    
    # re_order the columns
    df = re_order(df).copy()
    
    
    # setting the lables for confirmed planets and false positives
    df['Label 1'] = df['disposition'] + "-" + df['p_disposition']
    
    # setting a label column depending on the combination of Label 1 created column
    df['Label'] = np.where(df['Label 1'].str.contains('CONFIRMED-CANDIDATE', na=False), 1, '')
    df['Label'] = np.where(df['Label 1'].str.contains('FALSE POSITIVE-FALSE POSITIVE', na=False), 0,df['Label'] )
    df['Label'] = np.where(df['Label 1'].str.contains('CONFIRMED-FALSE POSITIVE'), 0, df['Label'])
    df['Label'] = np.where(df['Label 1'].str.contains('FALSE POSITIVE-CANDIDATE'), 0, df['Label']) 
    
    
    
    # two duplicates of the dataframe one for 1's and one for 0's
    df_labeled_1 = df.copy()
    df_labeled_2 = df.copy()
    
    # filter out the two categories
    df_labeled_1 = df_labeled_1[df_labeled_1['Label'].str.contains('1', na=False)]
    df_labeled_2 = df_labeled_2[df_labeled_2['Label'] == 0]
    
    # concatenate back together for singel output
    df_labeled = pd.concat([df_labeled_1, df_labeled_2])
    
    return df_labeled


def clean_output():

    # list of columns to be dropped
    drop_list = ['ID','kepoi_name','disposition','p_disposition','Label 1']
    
    # column to be added to the start with move_cols method
    col_list = ['Label']
    
    # duplicated of the labeled dataframe
    df_reordered = relabel_data().copy()
    
    # method to move the labeled column to the first column for readability
    def move_cols(frame):
        for x in range(len(col_list)):
            col_to_move = frame.pop(col_list[x])
            frame.insert(x, col_list[x], col_to_move)
        
        return frame
    
    # move the column
    df_reordered = move_cols(df_reordered)  
    # drop columns from the drop list that arent needed for machine learning  
    df_reordered.drop(drop_list, axis=1, inplace=True)
    
    return df_reordered


# get data
df_output = clean_output()
# output to csv
df_output.to_csv('D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/labeled_data.csv', index=False)
    
    

