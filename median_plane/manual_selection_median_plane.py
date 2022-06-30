#import natsort
import os
import nibabel as nib
import numpy as np
#import dicom2nifti
import matplotlib.pyplot as plt
#from PIL import Image
import skimage
import skimage.measure
from scipy import ndimage
import sparse
import pickle as pkl

from shutil import copyfile
#from openpyxl import load_workbook
import pandas as pd
from sys import exit

##########
# Functions for timing like matlab
import time
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
# This will be the main function through which we define both tic() and toc()
def toc(text=None, tempBool=True):
    if text is not None:
        print(text)
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
def tic(text=None):
    # Records a time in TicToc, marks the beginning of a time interval
    toc(text=text, tempBool=False)
###########


def run_median_segmentation(input_info, image_info, root, path_nifti, save_results=True, troubleshooting_images=[False, False]):
    """Main funcion to run a ribs segmentation. 
    Reads a dicom/nifti and outputs centroid locations of the ribs (sternum not ommitted).
    
    Arguments:
        input_info:    list of strings for [cohort, subject, condition, path_dicom]
        root:          string of root directory
        path_nifti:    string of path to directory containing nifti files
        save_results:  boolean, whether to save the resultant sparse array of rib centroids (default True)
        troubleshooting_images: boolean, whether to generate intermediate images for troubleshooting (default False)
    
    Returns:
        (nothing)
    """
    
    # Setup
    [cohort, subject, condition, path_dicom] = input_info
    print("\tRunning median segmentation...\t", subject, end="\n")
    threshold_cort = 1200     # Threshold for cortical bone  
    threshold_lung = -320    # Threshold for lung

    # Define the filepaths
    if cohort == "Human_Lung_Atlas":
        path_dicom_specific = os.path.join(path_dicom, cohort, subject, condition, "Raw", "")
    elif cohort == "Human_Aging":
        #  + "_oldformat"
        # , "Archive"
        path_dicom_specific = os.path.join(path_dicom, cohort, subject, condition, "Raw")

    # Check if the dicom directory exists and contains files, then run segmentation
    if not os.path.exists(path_dicom_specific):
        print("\tRunning median segmentation...\t", subject, "\tFailed. Directory not found: ", path_dicom_specific)
    elif len(os.listdir(path_dicom_specific) ) == 0:
        print("\tRunning median segmentation...\t", subject, "\tFailed. Directory is empty: ", path_dicom_specific)
    else:
        path_output = os.path.join(root, cohort, subject, condition, "Median")
        path_nifti_specific = os.path.join(path_nifti, cohort, subject, condition, "Torso", subject + ".nii")
        if not os.path.exists(path_output):
            os.makedirs(path_output)
               
        # If nifti doesn't already exist, convert dicom to nifti and load the nifti
        if not os.path.isfile(path_nifti_specific):
            convert_dicoms(subject, cohort, path_dicom_specific, path_nifti_specific)
        
        # Load the nifti, segment the ribs, and optionally save the results
        if os.path.isfile(path_nifti_specific):
            image = load_nifti(path_nifti_specific)
            # image = image_filter(image, intensity=6000)  # FILTER !!!
            
            # Plot slices to select coorindates that define median plane
            manually_select_median_points(image, subject, condition)

            print("\tRunning median segmentation...\t", subject, "\tDone.")



def convert_dicoms(subject, cohort, dicom_path, nifti_path):
    """Converts the dicoms files for a subject to a nifti file.
    """
      
    if cohort == "Human_Lung_Atlas":
        #dicom2nifti.common.is_slice_increment_inconsistent(dicom_path)
        # https://icometrix.github.io/dicom2nifti/readme.html?highlight=inconsistent
        # Disable the validation of the slice increment.
        # This allows for converting data where the slice increment is not consistent.
        # USE WITH CAUTION!
        # dicom2nifti.settings.disable_validate_slice_increment()

        try:
            dicom2nifti.dicom_series_to_nifti(dicom_path, nifti_path, reorient_nifti=True)   # changed to true
        except: # dicom2nifti.exceptions.ConversionValidationError:
            # subprocess.run(["plastimatch", "convert", "--input", dicom_path, "--output-img", nifti_path])
            #print("(", subject, "NIfTI error)", end=None)
            print("\tRunning torso segmentation...\t", subject, "\tFailed. NIfTI conversion error.")
            pass

    if cohort == "Human_Aging":
        try:
            dicom2nifti.dicom_series_to_nifti(dicom_path, nifti_path, reorient_nifti=True)
        except: # dicom2nifti.exceptions.ConversionValidationError:
            # subprocess.run(["plastimatch", "convert", "--input", dicom_path, "--output-img", nifti_path])
            #print("(", subject, "NIfTI error)", end=None)
            print("\tRunning torso segmentation...\t", subject, "\tFailed. NIfTI conversion error.")
            pass


###
def load_nifti(nifti_path):
    """Loads a nifti file given a specified path.
    """
    ct_im_nii = nib.load(nifti_path)
    ct_im = np.asarray(ct_im_nii.get_fdata())  # note: changed to get_fdata to prevent deprecation warn
    return ct_im
    
      
###
def manually_select_median_points(image, subject, protocol):

    image = np.moveaxis(image, 2, 0) # switch axes to enumerate easily (ORIGINAL -> AXIAL)
    
    # define slices to take
    slice_indices = [int(i) for i in np.linspace(.125, .875, 4)*len(image)]
    
    for i, axial_slice in enumerate(image[slice_indices]):
    
        plt.close('all')
        plt.figure()
        plt.imshow(axial_slice)
        plt.title(subject+" "+protocol+", z="+str(slice_indices[i]))
        print("\tShowing... "+subject+" "+protocol+", z="+str(slice_indices[i]))
        plt.show()
    




#############################################################################################################

#dicom_path = "/eresearch/lung/mpag253/Archive"
root = "/hpc/mpag253/Ribs/median_plane"
path_nifti = "/hpc/mpag253/Torso/segmentation"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/median_plane/points_for_median_plane.xlsx", skiprows=0, usecols=range(11)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_median_segmentation(input_list[i, 1:5], input_list[i, 5:], root, path_nifti, save_results=True,
                               troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")







