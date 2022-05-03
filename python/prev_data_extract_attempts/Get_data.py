# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:49:14 2021

@author: Jonathan
"""

import pandas as pd
from lightkurve import search_targetpixelfile
import lightkurve as lk
from tqdm import tqdm
import time


# method to import all needed data to retrieve Pixel Target Files, Star, Confirmed and Object of Interest Listings
def import_data():
    
    # Return a single planetary solution for confirmed planets with all columns from the Planetary Systems Composite Parameters Table. Output is in csv format.
    single_planetary_solutions = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv")
    
    # All Kepler Stellar, which includes Q1-12, Q1-16, Q1-17 DR 24, and Q1-17 DR 25 (All Stars and KEPLER ID)
    all_kepler_stars = ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=keplerstellar")
    
    # Kepler Objects of Interest
    all_kepler_objects_of_interest= ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative")
    
    single_planetary_solutions_df = pd.read_csv(single_planetary_solutions, low_memory=False)
    all_kepler_stars_df = pd.read_csv(all_kepler_stars, low_memory=False)
    all_kepler_objects_of_interest_df = pd.read_csv(all_kepler_objects_of_interest, low_memory=False)


    return single_planetary_solutions_df, all_kepler_stars_df, all_kepler_objects_of_interest_df

# create data frames for each download
data = import_data()
confirmed_df = data[0].copy()
all_stars_df = data[1].copy()
koi_df = data[2].copy()

# clean unneeded data
del data

#create empty array
li = []

def get_pixel_files():
    
    # copy of the kepler objects of interest
    koi_pixel_id_df = koi_df.copy()

    
    # for each item id in the list download the pixel file and covert to lightcurve
    for x in tqdm(range(len(koi_pixel_id_df)), "Getting Pixel Data..."):
        # get the latest quarter for the pixel data
        search_result = lk.search_lightcurve(f'KIC {koi_pixel_id_df.kepid[x]}', author='Kepler')
        most_recent = len(search_result.mission)
        details = search_result.mission[most_recent-1]
        qtr = details[-2:]
        qtr = int(qtr)
        try:
            # download the target pixel and convert to lightcurve
            tpf = search_targetpixelfile(f'KIC {koi_pixel_id_df.kepid[x]}', author="Kepler",quarter=qtr, cadence="long").download()
            lc =  tpf.to_lightcurve(aperture_mask='all')
            # remove nans, flatten and apply fold based on period
            lc = lc.remove_nans().flatten(window_length=401).fold(period=koi_pixel_id_df['koi_period'][x]).bin(time_bin_size=0.01)
            
            # remove the appereture mask and convert the light curve to pandas data frame 
            lc =  tpf.to_lightcurve(aperture_mask='all')
            df = lc.to_pandas()
            
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
            df['ID'] = lc.meta['LABEL']
            
            # add further information for later cleaning and classification
            df['kepoi_name'] = koi_pixel_id_df['kepoi_name'][x]
            df['disposition'] = koi_pixel_id_df['koi_disposition'][x]
            df['p_disposition'] = koi_pixel_id_df['koi_pdisposition'][x]
            df['period'] = koi_pixel_id_df['koi_period'][x]
            df['period_err1'] = koi_pixel_id_df['koi_period_err1'][x]
            df['period_err2'] = koi_pixel_id_df['koi_period_err2'][x]
            df['kepler_name'] = koi_pixel_id_df['kepler_name'][x]
        except AttributeError:
            continue
        

        # append to the list
        li.append(df)
        #time.sleep(1)
        
        #create a data frame and concatenate the array to it, creating one large data frame of all the files
    try:
        frame = pd.concat(li, axis=0, ignore_index=True)
    except ValueError:
        print("Error creating DataFrame, Exiting.")
        raise SystemExit

    return frame

# collect the target pixel information
df = get_pixel_files()

"""
    Because some stars have a much greater number of flux readings than others
    The will remove any flux columns that have greater than 80% nan values
    Avoiding removing the kepler name column and only emply flux columns
    Flux columns will have a minimum amount.
    
    Because the downloaded pixel files are "long" cadence the time interval betweemn
    exposures is all 30mins.
    
"""
perc = 80.0 # Like N %
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
df = df.dropna( axis=1, thresh=min_count)



# move these columns to the start or the dataframe for ease of reading later
col_list = ['ID','kepoi_name','disposition','p_disposition','period','period_err1','period_err2','kepler_name']
df_reordered = df.copy()


def move_cols(frame):
    for x in range(len(col_list)):
        col_to_move = frame.pop(col_list[x])
        frame.insert(x, col_list[x], col_to_move)
    
    return frame

df_reordered = move_cols(df_reordered)    


# export to csv
df_reordered.to_csv('./candidates/candidates_with_flux.csv', index=False)
koi_df.to_csv('./candidates/candidate_details.csv', index=False)
