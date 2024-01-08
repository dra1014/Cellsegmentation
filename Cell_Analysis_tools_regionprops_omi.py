from pathlib import Path
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
#C:\Users\jriendeau\Anaconda3\envs\seg\Lib\site-packages\cell_analysis_tools
from cell_analysis_tools.io import load_image
from cell_analysis_tools.flim import regionprops_omi



#%% This should be the only section where you should need to make changes

path_dataset = Path(r'Z:\skala\Andy\Danielle_Palecek_Pilote\Pilot\2022-11-02 Day 16')######################################################################################################

file_suffixes = {
    
    'mask': '_Ch2_000001.ome_cellpose.tiff',##################################################################################################################################
    
    'photons': '_photons.asc',
    'a1[%]': '_a1\[%\].asc',
    'a2[%]': '_a2\[%\].asc',
    't1': '_t1.asc',
    't2': '_t2.asc',
    }

#%% This finds your file paths for EACH image so you don't have to input everything
standard_dictionary = {
    # Mask file
    "mask" : "",
    
    # NADH files
    "nadh_photons" : "",
    "nadh_a1" : "",
    "nadh_a2" : "",
    "nadh_t1" : "",
    "nadh_t2" : "",
    
    # FAD files
    "fad_photons" : "",
    "fad_a1" : "",
    "fad_a2" : "",
    "fad_t1" : "",
    "fad_t2" : "",
    }
    
# GET LIST OF ALL FILES FOR REGEX
list_all_files = list(path_dataset.rglob("*"))
list_str_all_files = [str(b) for b in list_all_files]
    
list_all_nadh_photons_images = list(filter(re.compile("n" + file_suffixes["photons"]).search, list_str_all_files ))##############################################################
    
dict_dir = {}
for path_str_im_photons in tqdm(list_all_nadh_photons_images, desc='Assembling file dictionary'):
    pass
    
    # generate dict name
    path_im_photons_nadh = Path(path_str_im_photons)
    handle_im = path_im_photons_nadh.stem.rsplit('n', 2)[0]
    dict_dir[handle_im] = standard_dictionary.copy()
        
    # NADH 
    handle_nadh = handle_im + "n"
                
    # paths to NADH files
    dict_dir[handle_im]["nadh_photons"] = list(filter(re.compile(handle_nadh + file_suffixes['photons']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_a1"] = list(filter(re.compile(handle_nadh  + file_suffixes['a1[%]']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_a2"] = list(filter(re.compile(handle_nadh + file_suffixes['a2[%]']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_t1"] = list(filter(re.compile(handle_nadh + file_suffixes['t1']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_t2"] = list(filter(re.compile(handle_nadh + file_suffixes['t2']).search, list_str_all_files))[0]
    
    # MASKS
    try:
        dict_dir[handle_im]["mask"] = list(filter(re.compile(handle_nadh +  file_suffixes['mask']).search, list_str_all_files))[0]
    except IndexError:
        print(f"{handle_im} | mask missing")    
        del dict_dir[handle_im]
        continue
            
    # locate corresponding FAD photons image
    # try:
    #     path_str_im_photons_fad = list(filter(re.compile(handle_im + "f" + file_suffixes["photons"]).search, list_str_all_files))[0] ###################################
    # except IndexError:
    #     print(f"{handle_im} | FAD files missing")
    #     del dict_dir[handle_im]
    #     continue
        
    # path_im_photons_fad = Path(path_str_im_photons_fad)
    # handle_fad = handle_im + "f"

    # # paths to FAD files
    # dict_dir[handle_im]["fad_photons"] = list(filter(re.compile(handle_fad +  file_suffixes['photons']).search, list_str_all_files))[0]
    # dict_dir[handle_im]["fad_a1"] = list(filter(re.compile(handle_fad +  file_suffixes['a1[%]']).search, list_str_all_files))[0]
    # dict_dir[handle_im]["fad_a2"] = list(filter(re.compile(handle_fad +  file_suffixes['a2[%]']).search, list_str_all_files))[0]
    # dict_dir[handle_im]["fad_t1"] = list(filter(re.compile(handle_fad +  file_suffixes['t1']).search, list_str_all_files))[0]
    # dict_dir[handle_im]["fad_t2"] = list(filter(re.compile(handle_fad +  file_suffixes['t2']).search, list_str_all_files))[0]
        
    df_paths = pd.DataFrame(dict_dir).transpose()
    df_paths.index.name = "base"



#%% load csv dicts with path sets 
import matplotlib.pyplot as plt     
# iterate through rows(image sets) in dataframe,
outputs = pd.DataFrame()
for base, row_data in tqdm(list(df_paths.iterrows()), desc='Analyzing images'): # iterate through sets in csv file
            pass

            # load mask image
            label_image = load_image(Path(str(row_data['mask'])))
            
            # load NADH images 
            im_nadh_intensity = load_image(Path(row_data.nadh_photons))
            im_nadh_a1 = load_image(Path(row_data.nadh_a1)); im_nadh_a1 = np.ma.masked_array(im_nadh_a1, mask=im_nadh_a1==0)
            im_nadh_a2 = load_image(Path(row_data.nadh_a2)); im_nadh_a2 = np.ma.masked_array(im_nadh_a2, mask=im_nadh_a2==0)
            im_nadh_t1 = load_image(Path(row_data.nadh_t1)); im_nadh_t1 = np.ma.masked_array(im_nadh_t1, mask=im_nadh_t1==0)
            im_nadh_t2 = load_image(Path(row_data.nadh_t2)); im_nadh_t2 = np.ma.masked_array(im_nadh_t2, mask=im_nadh_t2==0)
            # load NADH images 
            # im_fad_intensity = load_image(Path(row_data.fad_photons))
            # im_fad_a1 = load_image(Path(row_data.fad_a1)); im_fad_a1 = np.ma.masked_array(im_fad_a1, mask=im_fad_a1==0)
            # im_fad_a2 = load_image(Path(row_data.fad_a2)); im_fad_a2 = np.ma.masked_array(im_fad_a2, mask=im_fad_a2==0)
            # im_fad_t1 = load_image(Path(row_data.fad_t1)); im_fad_t1 = np.ma.masked_array(im_fad_t1, mask=im_fad_t1==0)
            # im_fad_t2 = load_image(Path(row_data.fad_t2)); im_fad_t2 = np.ma.masked_array(im_fad_t2, mask=im_fad_t2==0)
            
            

            # compute ROI props
            omi_props = regionprops_omi(
                image_id = base,
                label_image = label_image,
                im_nadh_intensity = im_nadh_intensity,
                im_nadh_a1 = im_nadh_a1, 
                im_nadh_a2 = im_nadh_a2, 
                im_nadh_t1 = im_nadh_t1, 
                im_nadh_t2 = im_nadh_t2,
                # im_fad_intensity = im_fad_intensity,
                # im_fad_a1 = im_fad_a1,
                # im_fad_a2 = im_fad_a2,
                # im_fad_t1 = im_fad_t1,
                # im_fad_t2 = im_fad_t2,
                other_props=['area', 'area_bbox', 'area_convex', 'solidity', 'euler_number', 'area_convex', 'eccentricity', 'axis_major_length', 'axis_minor_length', 'extent', 'feret_diameter_max', 'equivalent_diameter_area', 'orientation', 'perimeter', 'perimeter_crofton'] #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
                )
     
            #create dataframe
            df = pd.DataFrame(omi_props).transpose()
            df.index.name = "base"
            
            # add other dictionary data to df
            df["base"] = base
            for item_key in row_data.keys():
                df[item_key] = row_data[item_key]

            # combine all image data into one csv
            outputs = pd.concat([outputs, df], axis=0)
            pass
#%% Final df manipulations before export
if 'stdev' in outputs.columns:
    stdev_columns = [col for col in df.columns if 'stdev' in col.lower()] 
    outputs.drop(columns=stdev_columns, inplace=True) #removes stdev columns
    pass
elif 'weighted' in outputs.columns:
    weighted_columns = [col for col in df.columns if 'weighted' in col.lower()]
    outputs.drop(columns=weighted_columns, inplace=True) #removes intensity weighted columns
    pass

# remove path files from outputs csv
if 'mask' in outputs.columns:
    outputs = outputs.iloc[:,:outputs.columns.get_loc("mask")]

# finally.. export data
outputs.to_csv(path_dataset/ f"{path_dataset.stem}_features.csv")
