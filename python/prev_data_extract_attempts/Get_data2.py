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
     
    # Kepler Objects of Interest API endpoint query
    all_kepler_objects_of_interest= ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative")
    
    # read in the data 
    all_kepler_objects_of_interest_df = pd.read_csv(all_kepler_objects_of_interest, low_memory=False)


    return  all_kepler_objects_of_interest_df

# create data frames for candidate data
data = import_data()
koi_df = data.copy()

# clean unneeded variable from memory
del data

#create empty array
li = []


def get_pixel_files(largest, q):
    
    # copy of the kepler objects of interest
    koi_pixel_id_df = koi_df.copy()
    
    # settings for finding missions with most data

    
    # for each item id in the list download the pixel file and convert to lightcurve
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
            
            # apply changes to lightcurve data
            '''
                aperature mask - for removing dead spots and only concentrating on light filled pixels,
                (not applied) flatten - flatten the curve along the time scale axis
                nans - remove any non numeric data
                outliers - remove any outlier data such as infinite value floats and anything above upper bounds of 3 std dev
               (not applied) normalize - normalize (unscaled) further normalization will happen later
            '''
            lc =  tpf.to_lightcurve(aperture_mask='default')
            # lc = lc.flatten()
            lc = lc.remove_nans()
            lc = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper =3)
            # lc = lc.normalize(unit='unscaled')
            
            # convert the light curve to pandas data frame 
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
            
        except AttributeError:
            continue
        

        # append to the list
        li.append(df)
        time.sleep(1)
        
        #create a data frame and concatenate the array to it, creating one large data frame of all the files
    try:
        frame = pd.concat(li, axis=0, ignore_index=True)
    except ValueError:
        print("Error creating DataFrame, Exiting.")
        raise SystemExit

    return frame

# collect the target pixel information
df = get_pixel_files(largest, q)

# =============================================================================
# """
#     Because some stars have a much greater number of flux readings than others
#     The will remove any flux columns that have greater than 80% nan values
#     Avoiding removing the kepler name column and only emply flux columns
#     Flux columns will have a minimum amount.
#     
#     Because the downloaded pixel files are "long" cadence the time interval betweemn
#     exposures is all 30mins.
#     
# """
# perc = 80.0 
# min_count =  int(((100-perc)/100)*df.shape[0] + 1)
# df = df.dropna( axis=1, thresh=min_count)
# =============================================================================



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
df_reordered.to_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Outputs/candidates_with_flux.csv', index=False)
koi_df.to_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Outputs/candidate_details.csv', index=False)

