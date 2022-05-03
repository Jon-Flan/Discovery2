# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""

import pandas as pd # to read in the kepler id's from the object of interest file
import lightkurve as lk # lightcurve for accessing the astropy and MAST api
from tqdm import tqdm # for output the console a progress bar
import glob # for checking existing files in folder
import os # imported to clear cache
import shutil # imported to clear cache



"""
    File 1 
    
    This file imports the kepler objects of interest table that was previously downloaded 
    and uses the kepler ID to download all available lightcurves for that star
    
    Only stars that have a confirmed exoplanet or confirmed false positive are downloaded
    
    The entire 4 year period of all exposures are downloaded and stitched into one continous 
    Lightcurve and saved under the kepler ID for further analysis and pre processing before 
    being used in the Neural Networks that are to be tested.
    
    Imported data is approx 100GB after transfor the estimated total output is approx 30GB 
    
    7504 stars (removing duplicates gives 6610) - avg 50k images with flux readings per star
    For a total of approx 300 million images processed and flux readings stitched
    
    Total Stitched FLux Files = 6610 files 31.3GB
    
"""

# method to import all needed data to retrieve Lightcurve Files, Star, Confirmed and Object of Interest Listings
def import_data():
    
         
    # Kepler Objects of Interest file
    all_kepler_objects_of_interest= ("../data/candidates/candidate_details.csv")
    
    # read in the data 
    kepler_df = pd.read_csv(all_kepler_objects_of_interest, low_memory=False)
    
    # only keep confirmed or negative instances
    kepler_df = kepler_df[kepler_df['koi_disposition'].str.contains('CONFIRMED') | 
                          kepler_df['koi_disposition'].str.contains('FALSE')]

    
    # reset the index 
    kepler_df.reset_index(drop=True, inplace=True)
    
    return  kepler_df


# check the folder for any already saved FITS files and remove from the list to download
def create_list():
    # path to fits files
    path = (r'D:/College/year_4/semester_2/Software_project/Discovery2/data/Fits')
    
    # use glob package to create list of files
    all_files = glob.glob(path+"\*")
    
    # remove the path extension and convert all id's to integers
    files = [sub.replace('D:/College/year_4/semester_2/Software_project/Discovery2/data/Fits\\', '') for sub in all_files]
    files = [int(i) for i in files]
    
    # import the kepler candidate info
    kepler_list = import_data().copy()
    
    # filter the kepler dataframe removing any already downloaded fits files
    kepler_list = kepler_list[~kepler_list['kepid'].isin(files)]
    kepler_list.reset_index(drop=True, inplace=True)
    
    # return new dataframe
    return kepler_list    
    
    
# download and stitch all relevant lightcurve to export to FITS files for each star
def get_light_curves():
    
    # copy of the kepler objects of interest
    koi_lc_df = create_list().copy()
    

    # for each item id in the list download all lightcurves available and stitch together
    for x in tqdm(range(len(koi_lc_df)), "Getting Data..."):
        try:
             # get all light curve data for particular star
            search_result = lk.search_lightcurve(f'KIC {koi_lc_df.kepid[x]}', author='Kepler').download_all()
            
            # stich all the light curves together usually > 65K
            lc_raw = search_result.stitch()
            
            # export to FITS file
            lc_raw.to_fits(path=f'D:\\College\\year_4\\semester_2\\Software_project\\Discovery2\\data\\Fits\\{koi_lc_df.kepid[x]}',
                              overwrite=False, flux_column_name='FLUX')
            
            
            
            # clear local cache after every download
            folder = ('C:/Users/Jonathan/.lightkurve-cache/mastDownload/Kepler/')
            
            # for each file name in the folder check if file or folder and delete
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        # if OS Error then ignore and continue
        except OSError as err:
            print(err)
            continue
            
    return 


# main function
def main():
    
    # var for number of attempts and success
    attempt = 1
    success = False
    
    # while success is false download the lightcurves
    while(success == False):
        print("Attempt: " + str(attempt))
        # download the lightcurves
        try:
            get_light_curves()
            success = True
        # if theres a break in connection reattempt
        except ConnectionError as err1:
            print(err1)
            attempt = attempt +1

# calling main function.
if __name__ == "__main__":
    main()




  
    
  
