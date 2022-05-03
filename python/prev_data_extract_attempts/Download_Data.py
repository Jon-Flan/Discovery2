# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 20:51:32 2022

@author: Jonathan
"""

import pandas as pd
import numpy as np
import lightkurve as lk
from tqdm import tqdm
import time




# method to import all needed data to retrieve Pixel Target Files, Star, Confirmed and Object of Interest Listings
def import_data():
    
    # get prev downloaded data
    prev = (r'D:/College/year_4/semester_1/Software_project/Discovery2/data/Outputs/candidates_with_flux1.csv')
    prev_df = pd.read_csv(prev)
    
    # convert the ID to an int
    prev_df['ID'] = prev_df['ID'].str.replace("KIC ", "")
    prev_df['ID'] = prev_df['ID'].astype('int64')

    # create a list of the IDs
    prev_ids = list(prev_df.ID)
         
    # Kepler Objects of Interest file
    all_kepler_objects_of_interest= ("D:/College/year_4/semester_1/Software_project/Discovery2/data/candidates/candidate_details.csv")
    
    # read in the data 
    kepler_df = pd.read_csv(all_kepler_objects_of_interest, low_memory=False)
    
    # only keep confirmed or negative instances
    kepler_df = kepler_df[kepler_df['koi_disposition'].str.contains('CONFIRMED') | 
                          kepler_df['koi_disposition'].str.contains('FALSE')]
    
    # filter out prev downloaded kep id's
    kepler_df = (kepler_df[~kepler_df.kepid.isin(prev_ids)])

    # reset the index 
    kepler_df.reset_index(drop=True, inplace=True)
    
    return  kepler_df

# create data frames for candidate data
data = import_data()
koi_df = data.copy()


# clean unneeded variable from memory
del data

#create empty array
li = []

def get_light_curves():
    
    # copy of the kepler objects of interest
    koi_lc_df = koi_df.copy()
    
    # settings for finding missions with most data

    
    # for each item id in the list download the pixel file and convert to lightcurve
    for x in tqdm(range(len(koi_lc_df)), "Getting Data..."):
        try:
             # get all light curve data for particular star
            search_result = lk.search_lightcurve(f'KIC {koi_lc_df.kepid[x]}', author='Kepler').download_all()
            
            # apply period, t0 and duration from objects table
            period, t0, duration_hours = koi_lc_df.koi_period[x], koi_lc_df.koi_time0bk[x], koi_lc_df.koi_duration[x]
            
            # stich all the light curves together > 65K
            lc_raw = search_result.stitch()
            
            # remove outliers based on lower level inf value and 3 std dev for upper bounds
            lc_clean = lc_raw.remove_outliers(sigma_lower=float('inf'), sigma_upper =3)
            
            # apply a temporary fold based on the period and t0 values
            temp_fold = lc_clean.fold(period, epoch_time=t0)
            
            # get the fractional duration based on days and the period 
            fractional_duration = (duration_hours / 24.0) / period
            
            # apply a phase masking of the temp fold less the 1.5 the fractional duration and make an abs number
            phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
            
            # apply a transit masking from the time value in flux file and the temp folding phase
            transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])
            
            # flatten, fold and apply mask to the light curve to reduce the values to approx 1.5K
            lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)
            lc_fold = lc_flat.fold(period, epoch_time=t0)
    
            # get the global flux and create bin sizes then normalize for ML input later
            lc_global = lc_fold.bin(bins=2000).normalize() - 1
            
            # remove any nans from the new flux generated
            lc_global = lc_global.remove_nans()
            lc_global = (lc_global / np.abs(lc_global.flux.min()) ) * 2.0 + 1
    
    
            # convert the light curve to pandas data frame 
            df = lc_global.to_pandas()
                
            # insert an incrimented column and convert to string
            df.insert(0, 'Flux_', range(1, 1+len(df)))
            df['Flux_'] = df['Flux_'].astype(str) 
            # add increment to combined column
            df['Flux'] = 'Flux ' + df['Flux_']
            
            # strip out only the flux reading
            df = df.reindex(columns=['Flux', 'flux'])
            
            # transpose the column to a row and reset the index
            df = df.set_index('Flux').T
            df.reset_index(inplace=True)
            
            # rename the column for the Kepler ID
            df.rename(columns={'index':'ID'}, inplace=True)
            df['ID'] = lc_global.meta['LABEL']
            
            # add further information for later cleaning and classification
            df['kepoi_name'] = koi_lc_df['kepoi_name'][x]
            df['disposition'] = koi_lc_df['koi_disposition'][x]
            df['p_disposition'] = koi_lc_df['koi_pdisposition'][x]
            
            # append to the list
            li.append(df)
            time.sleep(1)
        except ValueError:
            continue
        
        
    #create a data frame and concatenate the array to it, creating one large data frame of all the files
    try:
        frame = pd.concat(li, axis=0, ignore_index=True)
    except ValueError:
        print("Error creating DataFrame, Exiting.")
        raise SystemExit
    
    # return the dataframe    
    return frame



# collect the target pixel information
df = get_light_curves()

# move these columns to the start or the dataframe for ease of reading later
col_list = ['ID','kepoi_name','disposition','p_disposition']
df_reordered = df.copy()


def move_cols(frame):
    for x in range(len(col_list)):
        col_to_move = frame.pop(col_list[x])
        frame.insert(x, col_list[x], col_to_move)
    
    return frame

df_reordered = move_cols(df_reordered)    


# export to csv
prev1 = (r'D:/College/year_4/semester_1/Software_project/Discovery2/data/Outputs/candidates_with_flux1.csv')
prev_df1 = pd.read_csv(prev1)

df2 = pd.concat([prev_df1, df])

df2.to_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Outputs/candidates_with_flux2.csv', index=False)
#koi_df.to_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Outputs/candidate_details.csv', index=False)
  
    
  
