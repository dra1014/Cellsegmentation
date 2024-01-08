from pathlib import Path

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import tifffile

from natsort import natsorted
from cell_analysis_tools.visualization import compare_images

from sdt_read.read_bruker_sdt import read_sdt150
from sdt_read.read_wiscscan_sdt import read_sdt_wiscscan
import os

from helper import _write_sdt
import numpy as np
import re

from tqdm import tqdm

import subprocess
#%%

debug = False

# path_sdts = Path("./sdts")
# path_masks = Path("./masks")
# path_output = Path("./sdts_summed")
# path_sdts = Path(r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\ECG - Mohit_CASPI\211027_Panc1_10s-60s\Paired_1\256_1s_n")
# path_sdts = Path(r"\\skala-dv1.discovery.wisc.edu\ws\skala\Jeremiah\230706 lymphocytes in PBMCs\Kayvan Analysis\FAD")
# path_sdts = Path(r"E:\Analysis\2P_Fitting_PIXELvsROIvsCASPI\TSeries-07182019-0742-004\WLS\ROI\Summed1secSDTs")
# path_sdts = Path(r"C:\Users\ksamimi\Desktop\temp\SumTest")
# path_sdts = Path(r"C:\Users\ksamimi\Desktop\temp\1ch")
# path_sdts = Path(r"E:\Analysis\Dissociated_OV_ibidi_M\Combined2")
# path_sdts = Path(r"\\skala-dv1.discovery.wisc.edu\ws\skala\Kayvan\Retinal Data\2021_03_21_RetinaExp10_Dissociated_OV_editing\Untransduced\SUMMED_1040nm")
#path_sdts = Path(r"Z:\skala\Andy\20231206\Swabian\A6_NoStain\1_Contraction_15s-002")

#path_input = path_sdts / "Z:\skala\Andy\20231206\Swabian\A6_NoStain\1_Contraction_15s-002\SDT_Original"
#path_output = path_sdts / "Z:\skala\Andy\20231206\Swabian\A6_NoStain\1_Contraction_15s-002\SDT_Summed_Mito_Overphoton"
# generate a list of folders to process

path_sdts = Path(r"Z:\skala\Andy\20231206\Swabian\A6_NoStain\1_Contraction_15s-002\SDT_Original")
path_masks = Path(r"Z:\skala\Andy\20231206\Swabian\A6_NoStain\1_Contraction_15s-002\Mask\Mito\Clear")
path_output = Path(r"Z:\skala\Andy\20231206\Swabian\A6_NoStain\1_Contraction_15s-002\SDT_Summed_Mito")

# Generate a list of folders to process
list_path_sdts = [sdt for sdt in path_sdts.rglob("*.sdt") if sdt.is_file()]

# SDT files
list_masks_files = [str(p) for p in natsorted(path_masks.glob("*.tif"))]

# Iterate through the SDT files
for idx, path_sdt in tqdm(enumerate(list_path_sdts[:])):
    # Find mask
    base_name = path_sdt.stem
    path_mask = list(filter(re.compile(f".*{base_name}.*").search, list_masks_files))[0]

    if not Path(path_mask).exists():
        print(f"Mask not found for: {path_sdt.name}")
        continue
#list_path_sdts = [sdt for sdt in natsorted(path_input.rglob("*.sdt")) if sdt.is_file()]

# list_masks_files = [str(p) for p in natsorted(path_input.glob("*.tiff"))]
#list_masks_files = [str(masktiff) for masktiff in natsorted(path_input.rglob("*.tif*")) if masktiff.is_file()]

# iterate throught the SDT files
#for idx, path_sdt in tqdm(enumerate(list_path_sdts[:])):
   # pass
    
    # Find mask
    #base_name = path_sdt.stem
    # path_mask = list(filter(re.compile(f".*{base_name}.*").search, list_masks_files))[0]
    # path_mask = list(filter(re.compile(f".*{base_name}.*mask.*").search, list_masks_files))[0]
    #path_mask = list(filter(re.compile(f".*{base_name}.*MitoMask.*").search, list_masks_files))[0]
    # path_mask = list(filter(re.compile(f".*{base_name}.*cellpose.*").search, list_masks_files))[0]
    

    #if not Path(path_mask).exists():
    #    print(f"Mask not found for : {path_sdt.name}")
     #   continue
        
    # load mask
    labels = tifffile.imread(path_mask)
   
    ############## Jenu's code to load SDT's
    # if os.path.getsize(path_sdt) > (2**25): # (file size is equal to 33555190, but ~32 MB is a good marker)
    # 	im = read_sdt_wiscscan(path_sdt)
    # else:
    # 	im = read_sdt150(path_sdt)
    
    # im = read_sdt_wiscscan(path_sdt)
    # if len(im.shape)==4:  # if SDT is multichannel
    #     im = im[:,:,:,2]  #  Ch1=0, Ch2=1, Ch3=2
    
    im = read_sdt150(path_sdt)    
    # if len(im.shape)==4:  # if SDT is multichannel
        # im = im[1,:,:,:]  #  Ch1=0, Ch2=1, Ch3=2
    if len(im.shape)==3:  # if SDT is single-channel
        im = im[np.newaxis, :]
    
    ############## Strip the header from the original SDT
    args = ["SDTZip.exe", "-uz", str(path_sdt)]
    subprocess.run(args)
    
    path_uncompressed_file = path_sdt.parent / path_sdt.name.replace(".", ".uncompressed.")
    while not path_uncompressed_file.exists():
        pass # wait while file is being created
    
    with open(path_uncompressed_file,'rb') as fid:
        # print(np.size(fid.read()))
        # binary_sdt = np.fromstring(fid.read(), np.uint16) 
        # binary_sdt = np.fromstring(fid.read(2*512*512*256+378), np.uint16) 
        binary_sdt = np.fromfile(fid, dtype=np.uint16)
    
    if not isinstance(im, np.ndarray):
        binary_data = im.to_numpy().astype(np.uint16)
    else:
        binary_data = im.astype(np.uint16)
    
    header_ = binary_sdt[0:np.size(binary_sdt)-np.size(binary_data)].tobytes()
    os.remove(path_uncompressed_file)
    ##############
    
    # if debug:
        # compare_images('sdt', im[0,:].sum(axis=2), "mask", labels)
    
    compare_images('sdt', im[0,:].sum(axis=2), "mask", labels)
    
    # placeholder array - start with 'float64' to avoid overflow. later convert to 'uint16'.
    sdt_decay_summed = (np.zeros_like(im)).astype(np.float64) 

    # iterate through labels
    list_labels = [l for l in np.unique(labels) if l != 0]
    for label in list_labels:
        pass
        mask_label = labels == label 
        
        # decay_roi = im * mask_label[...,np.newaxis] # mask decays
        decay_roi = im * mask_label[np.newaxis, ..., np.newaxis] # mask decays
        
        # decay_summed = decay_roi.sum(axis=(0,1)) 
        decay_summed = decay_roi.sum(axis=(1,2)) 
        
        if debug:

            fig, ax = plt.subplots(1,2, figsize=(10,5))
            fig.suptitle(f"{path_sdt.name} | label: {label}")
            # ax[0].imshow(decay_roi.sum(axis=2))
            ax[0].imshow(decay_roi[0,:].sum(axis=2))
            ax[0].set_aspect('equal')
            ax[1].plot(decay_summed[0,:])
            plt.show()
            # ax[1].set_aspect('equal')

            # plt.title(f"{path_sdt.name} | label: {label}")
            # plt.imshow(decay_roi.sum(axis=2))
            # plt.show()
            
            # plt.title(f"label: {label}")
            # plt.plot(decay_summed)
            # plt.show()
        
        # sdt_decay_summed[mask_label] = decay_summed[np.newaxis, np.newaxis,:]
        sdt_decay_summed = sdt_decay_summed.transpose(1, 2, 3, 0)
        decay_summed = decay_summed.transpose(1, 0)
        sdt_decay_summed[mask_label] = decay_summed[np.newaxis, np.newaxis,:]
        sdt_decay_summed = sdt_decay_summed.transpose(3, 0, 1, 2)
        
        # test 512x512x256
        # temp_array = np.zeros((512,512,256))
        # temp_array[:256,256:,...] = im
        # temp_array[:256,:256,...] = im
        # temp_array[256:,256:,...] = sdt_decay_summed
        # temp_array[256:,:256,...] = sdt_decay_summed
        # _write_sdt(path_output / f"{path_sdt.stem}_summed_512_BH.sdt", 
        #             temp_array, 
        #             resolution=512, 
        #             manufacturer="BH")
        #####
    
    bincount_max = np.max(sdt_decay_summed);
    if bincount_max>65000:  # if peak count overflows the 16-bit variable, scale everything down
        sdt_decay_summed = (np.round(sdt_decay_summed * (65000/bincount_max))).astype(np.uint16)
    else:
        sdt_decay_summed = sdt_decay_summed.astype(np.uint16)
    
    
    print(path_output / f"{path_sdt.stem}_summed.sdt")
    num_ch, width, _, _ = im.shape
    path_file = path_output.absolute() / f"{path_sdt.stem}_summed.sdt" 
    # _write_sdt(path_file, sdt_decay_summed, resolution=width)
    # _write_sdt(path_file, sdt_decay_summed, manufacturer="BH", resolution=width)
    # combine header and data
    phantom_data = header_ + sdt_decay_summed.tobytes()
    
    with open(path_file,'wb') as fid:
        fid.write(phantom_data)   

    
    
    #### COMPRESS FILES
    # args = ["SDTZip.exe", "-z", str(path_file)]
    # subprocess.run(args)
    
    # path_compressed_file = path_file.parent / path_file.name.replace(".", ".compressed.")
    # while not path_compressed_file.exists():
    #     pass # wait while file is being created
    
    # # remove uncompressed file
    # os.remove(path_file)
    
    # # rename compressed file to uncompressed file
    # os.rename(path_compressed_file, path_file)
    
            
     
        
