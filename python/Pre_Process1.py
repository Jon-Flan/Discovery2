# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""

import pandas as pd # to read in the kepler id's from the object of interest file
import lightkurve as lk # lightcurve for accessing the astropy and MAST api
from tqdm import tqdm # for output the console a progress bar
import numpy as np # for cleaning / flattening and folding the lightcurve on its phase
from os.path import exists # for checking if a file exists

# import functions from Get_Data.py
import Get_Data as gd

"""
    File 2
    
    This file takes the downloaded lightcurves (FITS Files) and creates
    a pandas dataframe for each flux by using the period, time0bk, and duration
    values for each of the observations and folding the lightcurve on the correct phase,
    normalises is applied during the binning process,  each lightcurve is placed in an
    array of 2000 bins and nan is interolpated using linear function. 
    
    resulting output is transposed dataframe for each lightcurve consisting of
    the ID, the dispostion, the p disposition, the kepler name and 2000 light curve readings.
    
"""

# empty list to store all dataframes
li = []


# transpose the dataframe from a column to a row
def transpose_dataframe(df):
    
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
    
    return df


# add addiontal columns to the data for labelling later
def adjust_columns(df, koi_df, x, lc):
    
    # rename the column for the Kepler ID
    df.rename(columns={'index':'ID'}, inplace=True)
    df['ID'] = lc.meta['LABEL']
        
    # add further information for later cleaning and classification
    df['kepoi_name'] = koi_df['kepoi_name'][x]
    df['disposition'] = koi_df['koi_disposition'][x]
    df['p_disposition'] = koi_df['koi_pdisposition'][x]

    
    return df


# clean the lightcurve from noise using sigma upper and lower values
# fold the lightcurve on its designated period time and apply a phase mask to centre the tranist(if any)
def clean_flatten_fold(lc, period, t0, duration_hours):
    
    # clean outliers from the lightcurve
    lc_clean = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper =3)
          
    # create a temp fold on its designated phase to apply to the phase masking
    temp_fold = lc.fold(period, epoch_time=t0)
    
    # create phase mask based on the tranist period and peak values
    fractional_duration = (duration_hours / 24.0) / period
    phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
    transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])
    
    if False not in transit_mask:
        # flatten with adjusted phase mask
        fractional_duration = (duration_hours / 24.0) / period
        phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * -1.5)
        transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])

        lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)
    else:
        # further flatten the lightcurve, also returning the trend of noise that has been removed    
        lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)

    # further flatten the lightcurve, also returning the trend of noise that has been removed
    lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)
    
    # fold the light curve on its designated period 
    lc_fold = lc_flat.fold(period, epoch_time=t0)
    
    return lc_fold


# create equal sized bins for each lightcurve (2000) and normalize the data
def normalize_and_bin(lc):
    
    # bin each lightcurve into same sized bins and normalize    
    lc_global = lc.bin(bins=2000).normalize()-1
    
    # mask any nans and apply calculation
    lc_global_mask = ~np.isnan(lc_global.flux)
    lc_global = (lc_global / np.abs(lc_global[lc_global_mask].flux.min()) ) * 2.0 + 1
    
    return lc_global


# read in each FITS file and create to singular flux row in pandas dataframe
def create_dataframes():
    
    # import the kepler objects of interest
    koi_df = gd.import_data()
    
    for x in tqdm(range(len(koi_df)), "Getting Pixel Data..."):
        try:
            # read in the star flux readings
            lc = lk.read(f'D:/College/year_4/semester_2/Software_project/Discovery2/data/Fits/{koi_df["kepid"][x]}')
            # assign period, t0 and duration values
            period, t0, duration_hours = koi_df['koi_period'][x], koi_df['koi_time0bk'][x], koi_df['koi_duration'][x]
            
            # clean and fold the light curve
            lc = clean_flatten_fold(lc, period, t0, duration_hours)
            
            # normalize and bin each lightcurve
            lc = normalize_and_bin(lc)
            
            # convert the light curve to pandas data frame 
            df = lc.to_pandas()
            
            # interpolate missing values from lightcurves after binning
            df['flux'] = df['flux'].interpolate(method='slinear')
            
            # transpose the dataframe
            df = transpose_dataframe(df)
            
            # add info columns to dataframe
            df = adjust_columns(df, koi_df, x, lc)
            
            # append to the list
            li.append(df)
        except FileNotFoundError:
            continue


# main function
def main():
    
    # create the data frames and add to list    
    create_dataframes()
        
    # concatenate the list into one single dataframe
    frame = pd.concat(li, axis=0, ignore_index=True)
    
    # path to existing file
    path_to_file = "D:\College\year_4\semester_2\Software_project\Discovery2\data\Output\candidates_with_flux.csv"
    # return boolean for file eixting
    file_exists = exists(path_to_file)
    
    # if file exists, read it in and concatenate the dataframes for output
    if file_exists:
        df1 = pd.read_csv(path_to_file)
        output = pd.concat([df1, frame])
        output.to_csv(path_to_file, index=False)
    else:
        # export to csv 
        frame.to_csv(path_to_file, index=False)
    


# calling main function.
if __name__ == "__main__":
    main()
