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


def run_ribs_segmentation(input_info, root, path_nifti, save_results=True, troubleshooting_images=[False, False]):
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
    print("\tRunning ribs segmentation...\t", subject, end="\r")
    threshold_cort = 300     # Threshold for cortical bone  
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
        print("\tRunning torso segmentation...\t", subject, "\tFailed. Directory not found: ", path_dicom_specific)
    elif len(os.listdir(path_dicom_specific) ) == 0:
        print("\tRunning torso segmentation...\t", subject, "\tFailed. Directory is empty: ", path_dicom_specific)
    else:
        path_output = os.path.join(root, cohort, subject, condition, "Ribs")
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

            # Generate ... and save as sparse matrix
            ribs_image = generate_ribs_centroids(image, threshold_lung, threshold_cort, gen_images=troubleshooting_images)        
            # Save rib labels image to file
            if save_results:
                save_as_sparse(ribs_image, 'Rib_Labels/ribs_segmented_'+subject+'_'+condition+'.pkl')

            print("\tRunning ribs segmentation...\t", subject, "\tDone.")


def image_filter(image, intensity=0):
    """Apply filters to the image to better extract details.
    Used in development of torso segmentation and retained for potential future use.
    """

    im_cropped = np.copy(image)
    im_cropped[im_cropped > 0] = 0
    band_highpass = ndimage.gaussian_filter(im_cropped, 2)
    band_lowpass = ndimage.gaussian_filter(im_cropped, 1.5)  # 1.5
    bandpass = -(band_lowpass - band_highpass)
    bandpass_normd = (bandpass - np.amin(bandpass)) / (np.amax(bandpass) - np.amin(bandpass))
    bandpass_trans = np.power(bandpass_normd, 1.5)
    filter = -intensity*(bandpass_trans - np.median(bandpass_trans)) #-300*(bandpass_normd - np.median(bandpass_normd))
    im_filtered = image + filter

    # # Plot the images
    # fig, axs = plt.subplots(1, 3)
    # im1 = axs[0].imshow(image[400:700, 50:250, 0], vmin=np.amin(im_filtered), vmax=np.amax(im_filtered)) #[400:700, 50:250])
    # plt.colorbar(im1, ax=axs[0], orientation='horizontal')
    # im2 = axs[1].imshow(filter[400:700, 50:250, 0]) #[400:700, 50:250])
    # plt.colorbar(im2, ax=axs[1], orientation='horizontal')
    # im3 = axs[2].imshow(im_filtered[400:700, 50:250, 0], vmin=np.amin(im_filtered), vmax=np.amax(im_filtered))  # [400:700, 50:250])
    # plt.colorbar(im3, ax=axs[2], orientation='horizontal')
    # plt.show()

    # # Plot the images - thresholds
    # fig, axs = plt.subplots(1, 4)
    # threshold_1 = np.zeros(np.shape(image))
    # threshold_1[im_filtered > -320] = 1
    # threshold_2 = np.zeros(np.shape(image))
    # threshold_2[im_filtered > -150] = 1
    # threshold_3 = np.zeros(np.shape(image))
    # threshold_3[im_filtered > -0] = 1
    # im1 = axs[0].imshow(image[:, :, 0])
    # im1 = axs[0].imshow(threshold_1[:, :, 0])
    # im2 = axs[1].imshow(threshold_2[:, :, 0])
    # im3 = axs[2].imshow(threshold_3[:, :, 0])
    # plt.show()

    return im_filtered


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


def load_nifti(nifti_path):
    """Loads a nifti file given a specified path.
    """
    ct_im_nii = nib.load(nifti_path)
    ct_im = np.asarray(ct_im_nii.get_fdata())  # note: changed to get_fdata to prevent deprecation warn
    return ct_im
    
    
def generate_img_fig(binary_img, filename, print_string, generate_image):
    """Saves a specified image to file if 'generate_image' is True
    """
    if generate_image:
        plt.figure()
        plt.imshow(binary_img)
        plt.savefig(filename+".tiff")
        print("\t\tSaved... "+filename+".tiff: "+print_string)
        # Alternative option to show image figure for better detail
        #print("\t\tShowing... "+filename+".tiff: "+print_string)
        #plt.show()
        

def size_of_labels(im, bg=-1):
    """ Returns the number of voxels for each label in a labelled image, 'im'.
        Excludes the background specified by 'bg'.
    """
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    return vals, counts
        
        
def centroids_of_labels(im, bg=-1):
    """ Returns the centroid, in pixel indices, for each label in a labelled 2D image, 'im'.
        Excludes the background specified by 'bg'.
    """
    vals, counts = np.unique(im, return_counts=True)
    centroids = [[] for _ in range(len(vals))]
    for i in range(len(vals)):
        indices = np.nonzero(im==vals[i])
        centroids[i] = [int(np.mean(indices[0])), int(np.mean(indices[1]))]
    return centroids, vals
        
        
def centroids_of_labels_3d(im, bg=-1):
    """ Returns the centroid, in pixel indices, for each label in a labelled 3D image, 'im'.
        Excludes the background specified by 'bg'.
    """
    vals, counts = np.unique(im, return_counts=True)
    centroids = [[] for _ in range(len(vals))]
    for i in range(len(vals)):
        indices = np.nonzero(im==vals[i])
        centroids[i] = [int(np.mean(indices[0])), int(np.mean(indices[1])), int(np.mean(indices[2]))]
    return centroids, vals
    

def largest_label_volume(im, bg=-1):
    """ Returns the label with the most instances in a labelled 3D image, 'im'.
        Excludes the background specified by 'bg'.
    """
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
        

def largest_label_n_volumes(im, n_vols=2, bg=-1):
    """ Returns the n='n_vols' labels with the most instances in a labelled image, 'im'.
        Excludes the background specified by 'bg'.
    """
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    #print(n_vols)
    #print(len(counts))
    n_cnts = len(counts)
    if n_cnts > 0:
        if n_cnts < n_vols:                                              # mod
            if n_cnts == 1:                                              # mod
                #print(counts)
                return vals                                              # mod
            else:                                                        # mod
                return vals[np.argpartition(counts, -n_vols)[-n_cnts:]]  # mod
        else:                                                            # mode
            return vals[np.argpartition(counts, -n_vols)[-n_vols:]]      # original
    else:
        return None
        
        
def eliminate_points_by_theta(im, thetass, centre, bg=-1):
    """ In a 3D image, 'im', eliminate voxels outside of the angle ranges specified in 'thetass'.
        Iterates through axis 0 of the image and evaluates angles in the plane of axes 1-2.
        Voxels are eliminted by assigning the value specified by 'bg'.
        thetass: list of angle pairs that specify regions to retain in the image.
        centre: centre point at which to evaluate the angles from.
    """

    indices = np.indices([np.shape(im)[1], np.shape(im)[2]])      
    indices[0] = indices[0] - int(centre[0])
    indices[1] = indices[1] - int(centre[1])
    angles = np.arctan2(indices[0], indices[1])
    eliminate = np.zeros(np.shape(angles))
    for thetas in thetass:
        eliminate += ((angles<thetas[0])|(angles>thetas[1])).astype(int)
    
    eliminate = eliminate==len(thetass)
    #print("Eliminated voxels: ", sum(sum(eliminate)))
    #plt.close('all')
    #fig = plt.figure()
    #plt.spy(eliminate)
    #plt.show()
    
    #indices = np.nonzero(eliminate) 
    ##slow here!!!
    #for i in range(np.shape(im)[dim]):
    #    for j in range(len(indices[0])):
    #        im[i][indices[0][j]][indices[1][j]] = bg

    # improved way
    eliminate = np.tile(eliminate, [np.shape(im)[0], 1, 1])
    im[eliminate==1] = bg
          
    return im


###
def segment_lungs(image, threshold=-320, gen_images=[False, False]):
    """ Simple segmentation of the lungs from a CT image. Segments using axial slices. 
    """

    # Parameters for troubleshooting images
    [gen_im1, gen_im2] = gen_images
    if gen_im1:
        print("\n", end="\r")
    case_name = "troubleshooting_image_"
    smpl_slc = -1 #int(np.shape(image)[2]/2)  #-85 #
        
    # Threshold 3d image to create pseudo-binary image
    # not actually binary, but 1 and 2 (for air and torso, respectively)
    # 0 will be treated as background, which we do not want
    lung_image = np.array(image > threshold, dtype=np.int8) + 1  # air=1, torso=2; original threshold=-320 
    generate_img_fig(lung_image[:,:,smpl_slc], case_name+"lung-2", "thresholded", gen_im1)   

    # Keep only the largest solid structure (in each slice)
    lung_image = np.moveaxis(lung_image, -1, 0) # switch axes to enumerate easily
    for i, axial_slice in enumerate(lung_image):
        axial_slice = axial_slice - 1 # converts to pseudo-binary: air=0, torso=1
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        l_max = largest_label_volume(labels, bg=0)
        lung_image[i][labels != l_max] = 1  # air=1, torso=2
    generate_img_fig(lung_image[smpl_slc, :, :], case_name+"lung-3", "???", gen_im1)

    # Identifying the background (in each slice)
    for i, axial_slice in enumerate(lung_image):
        labels = skimage.measure.label(axial_slice)
        background_labels = np.unique( (labels[ 0,  :], labels[-1,  :], labels[ :, 0], labels[:, -1]) )
        lung_image[i][np.isin(labels, background_labels)] = 0  # bg=0, air=1, torso=2
    generate_img_fig(lung_image[smpl_slc,:,:], case_name+"lung-4", "identify background", gen_im1)

    # Keep only the two largest air regions in each slice
    for i, axial_slice in enumerate(lung_image):
        axial_slice[axial_slice == 2] = 0 # converts to pseudo-binary: air=1, other=0
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        l_maxs = largest_label_n_volumes(labels, n_vols=2, bg=0)
        for l_max in l_maxs:
            axial_slice[labels == l_max] = 2  # lung=2, air=1, other=0
        axial_slice[axial_slice > 0] -= 1
        lung_image[i] = axial_slice
    generate_img_fig(lung_image[smpl_slc, :, :], case_name+"lung-4b", "???", gen_im1)
    lung_image = np.moveaxis(lung_image, 0, -1)  # switch axes back to normal

    return lung_image


###
def generate_lung_measures(lung_image):
    """ Get lung centre and span (both in pixels) from a binary image of the lungs.
    """
    indices = np.nonzero(lung_image)
    mins = np.min(indices, axis=1)
    maxs = np.max(indices, axis=1)
    lungs_span = maxs-mins
    lungs_centre = ((maxs+mins)/2).astype(int)
    return lungs_centre, lungs_span


###
def generate_rib_region(lung_image, lungs_span):
    """ Generates a binary image that specifies the valid region for ribs in the 3D image.
        Specifically, this method uses proximity to the lungs to define the rib region. 
    """
       
    #Inputs
    # number of pixels for the proximity region outisde the lungs (independently)
    size_dilation = int(30/624*lungs_span[0]) #30/624 - optimised + generalised from AGING001   
    # number of pixels for the proximity region inside the convex hull of both lungs
    size_erosion = int(60/624*lungs_span[0])  #60/624 - optimised + generalised from AGING001
    #print(size_dilation, size_erosion)
    dilated_lung = ndimage.binary_dilation(lung_image, iterations=size_dilation).astype(lung_image.dtype)
    
    # Generate rib region
    # Only keep the bone label if the centroid is in proximity of the lungs
    # + Eroded convex hull of the lungs to remove interior
    lung_cvhull = np.zeros(np.shape(lung_image))
    for i, axial_slice in enumerate(lung_image):
        lung_cvhull[i, :, :] = skimage.morphology.convex_hull_image(lung_image[i, :, :])
    eroded_lung_cvhull = ndimage.binary_erosion(lung_cvhull, iterations=size_erosion).astype(lung_image.dtype)
    rib_region = dilated_lung - eroded_lung_cvhull
    rib_region[rib_region < 0] = 0

    ## Adding additional region for spinal processes
    ## used this to try and generate planes of symmetry
    #size_dilation_extra = 50
    #dilated_lung_extra = ndimage.binary_dilation(dilated_lung, iterations=size_dilation_extra).astype(lung_image.dtype)
    #print(np.unique(dilated_lung_extra))
    #indices = np.nonzero(1-dilated_lung_extra)
    #spine_region = np.zeros(np.shape(dilated_lung_extra))
    #print(np.shape(indices)[1])
    #print(lungs_centre)
    #for i in range(np.shape(indices)[1]):
    #    if (indices[1][i] < lungs_centre[0] +50):
    #        if (indices[1][i] > lungs_centre[0] -50):
    #            if indices[2][i] < lungs_centre[1]:
    #                spine_region[indices[0][i], indices[1][i], indices[2][i]] = 1
    #rib_region += spine_region.astype(int)

    return rib_region
      

###
def segment_ribs(image, rib_region, threshold=600, gen_images=[False, False]):
    """ Segments the provided image and returns an image of centroids of the ribs
        (i.e. the bony bodies within the specified rib region).
    
    Inputs:
        image:          numpy ndaray, from nifti image of the scan
        rib_region:     numpy ndaray, binary image with same shape as 'image', defines valid areas for the centroids of rib bodies
        threshold:      int, threshold value for cortical bone (default = 600)
        save_images:    boolean, whether to save images for validation/troublshooting     
        
    """

    # Parameters for troubleshooting images
    [gen_im1, gen_im2] = gen_images
    if gen_im1:
        print("\n", end="\r")
    case_name = "troubleshooting_image_"
    smpl_slc = -1 #int(np.shape(image)[2]/2)  #-85 #

    # Threshold 3d image to create pseudo-binary image
    # not actually binary, but 1 and 2 (for air and torso, respectively)
    # 0 will be treated as background, which we do not want
    generate_img_fig(image[smpl_slc,:,:], case_name+"1P", "initial image", gen_im1)
    generate_img_fig(rib_region[smpl_slc,:,:], case_name+"2P", "rib region", gen_im1)
    binary_image = np.array(image > threshold, dtype=np.int8) + 1  # soft/air=1, bone=2; original threshold=-320 
    generate_img_fig(binary_image[smpl_slc,:,:], case_name+"3P", "thresholded", gen_im1)  

    # Keep only the large solid structures (in each slice)
    for i, axial_slice in enumerate(binary_image):
        axial_slice = axial_slice - 1 # converts to pseudo-binary: soft/air=0, torso=1
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        l_vals, l_counts = size_of_labels(labels, bg=0)
        for j in range(len(l_vals)):
            if l_counts[j] > 50:    
               binary_image[i][labels == l_vals[j]] = 3  # soft/air=1, bone=2, large_bone=3
    binary_image[binary_image > 1] -= 1 # large_bone=2, rest=1
    generate_img_fig(binary_image[smpl_slc, :, :], case_name+"4P", "???", gen_im1)

    # Get the centroids of each bone label and retain centroids in proximity to lungs
    for i, axial_slice in enumerate(binary_image):
        axial_slice = axial_slice - 1 # converts to pseudo-binary: bone=1, rest=0
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        centroids, label_list = centroids_of_labels(labels, bg=0)
        retain_labels = []
        for j, centroid in enumerate(centroids): #centroids[i]:
            if rib_region[i][centroid[0]][centroid[1]] == 1:
                retain_labels.append(label_list[j]) 
                #print("Centroid in proximity to lung:", centroid)    
        #if i == smpl_slc:
        #    print("Retained labels from sample slice:", retain_labels)
        for label in label_list:
            if label not in retain_labels:
                binary_image[i][labels == label] = 1
    generate_img_fig(binary_image[smpl_slc, :, :], case_name+"5P", "retain ribs", gen_im1) 
    
    # Close and solidify remaining objects
    dilation_its = 5
    binary_image -= 1
    dilated_ribs = ndimage.binary_dilation(binary_image, iterations=dilation_its).astype(binary_image.dtype)
    generate_img_fig(dilated_ribs[smpl_slc, :, :], case_name+"6P", "dilate ribs", gen_im1)  
    binary_image += 1
    dilated_ribs += 1
    for i, axial_slice in enumerate(dilated_ribs):
        #axial_slice = axial_slice - 1 # converts to pseudo-binary: bone=1, rest=0
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        background_labels = np.unique( (labels[ 0,  0], labels[-1,  0], labels[-1, 0], labels[-1, -1]) )
        dilated_ribs[i][np.isin(labels, background_labels)] = 0  # bg=0, air=1, torso=2
    generate_img_fig(dilated_ribs[smpl_slc, :, :], case_name+"7P", "solidify ribs", gen_im1)
    dilated_ribs[dilated_ribs > 0] = 1
    eroded_ribs = ndimage.binary_erosion(dilated_ribs, iterations=dilation_its).astype(binary_image.dtype)
    generate_img_fig(eroded_ribs[smpl_slc, :, :], case_name+"8P", "erode ribs", gen_im1) 
    
    ## To generate ribs images for all slices...
    #for i in range(len(eroded_ribs)):
    #    generate_img_fig(lung_image[:, :, i]+2*eroded_ribs[i, :, :], "images/ribs_slice_"+"{:03d}".format(i), "slice image", gen_im1)
    #    plt.close('all')

    # Regenerate final rib centroids
    for i, axial_slice in enumerate(eroded_ribs):
        #axial_slice = axial_slice - 1 # converts to pseudo-binary: bone=1, rest=0
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        centroids, label_list = centroids_of_labels(labels, bg=0)
        binary_image[i] = 0
        for centroid in centroids[1:]:
            binary_image[i][centroid[0]][centroid[1]] = 1
            
    return binary_image
    
 
###   
def generate_ribs_centroids(image, threshold_lung, threshold_cort, gen_images=[False, False]):
  
    # Generate lung segmentation and extract measures of the lung
    tic()
    lung_image = segment_lungs(image, threshold=threshold_lung, gen_images=gen_images)     
    # generate measures from the lung image
    lungs_centre, lungs_span = generate_lung_measures(lung_image)
    #toc(text="Lung image + measures:")
            
    # Generate the rib region image
    lung_image = np.moveaxis(lung_image, -1, 0)     # switch axes to enumerate axial slices easily
    rib_region = generate_rib_region(lung_image, lungs_span)
    rib_region = np.moveaxis(rib_region, 0, -1) # revert axes to original
    #toc(text="Rib region:")
    
    # Generate POSTERIOR section: transverse slices
    # (original axes are in transverse slice configuration)
    image_posterior = segment_ribs(image, rib_region, threshold=threshold_cort, gen_images=gen_images)
    #toc(text="Anterior image:")

    ## Generate ANTERIOR section: axial OR coronal slices
    slice_direction = 'c'
    if slice_direction in ['c', 'coronal']:
        image = np.moveaxis(image, 0, 2) # switch axes to enumerate easily (ORIGINAL -> CORONAL)
        rib_region = np.moveaxis(rib_region, 0, 2) # switch axes to enumerate easily (ORIGINAL -> CORONAL)
        image_anterior = segment_ribs(image, rib_region, threshold=threshold_cort, gen_images=gen_images)  
        image_anterior = np.moveaxis(image_anterior, 2, 0) # revert axes (CORONAL -> ORIGINAL)
    elif slice_direction in ['a', 'axial']:
        image = np.moveaxis(image, 2, 0) # switch axes to enumerate easily (ORIGINAL -> AXIAL)
        rib_region = np.moveaxis(rib_region, 2, 0) # switch axes to enumerate easily (ORIGINAL -> AXIAL)
        image_anterior = segment_ribs(image, rib_region, threshold=threshold_cort, gen_images=gen_images)  
        image_anterior = np.moveaxis(image_anterior, 0, 2) # revert axes (AXIAL -> ORIGINAL)   
    #toc(text="Posterior image:")
    
    # Eliminate angular regions from anterior and posterior sections and merge
    # eliminate
    image_posterior = np.moveaxis(image_posterior, -1, 0) # switch axes to enumerate easily (ORIGINAL -> AXIAL)
    image_anterior = np.moveaxis(image_anterior, -1, 0) # switch axes to enumerate easily (ORIGINAL -> AXIAL)
    image_posterior = eliminate_points_by_theta(image_posterior, [[-1.1*np.pi, -0.66*np.pi], [0.66*np.pi, 1.1*np.pi]], lungs_centre[:2], bg=0)
    image_anterior = eliminate_points_by_theta(image_anterior, [[-0.66*np.pi, 0.66*np.pi],], lungs_centre[:2], bg=0)
    # merge
    binary_image = image_anterior + image_posterior
    binary_image = np.moveaxis(binary_image, 0, -1)
    #toc(text="Eliminate & merge:")
    
    # Generate labelled groups of objects
    # Dilate the ribs to (relatively) quickly establish connectivity, then label
    dilated_rib_centroids = ndimage.binary_dilation(binary_image, iterations=4).astype(binary_image.dtype)
    labels = skimage.measure.label(dilated_rib_centroids)
    # Apply labels to original (i.e. un-dilated) centroids
    labels = np.multiply(labels, binary_image)
    #toc(text="Combined and labelled:")

    # Remove labels with few occurences
    vals, counts = size_of_labels(labels, bg=0)
    for i in range(len(vals)):
        if counts[i] < int(20/624*lungs_span[0]):  # 20/624 based on AGING001
            labels[labels == vals[i]] = 0

    # Reset the label numbers
    unique_labels = np.unique(labels)
    rib_labels = np.zeros(np.shape(labels))
    for i, label in enumerate(unique_labels):
        rib_labels[labels==label] = i
       
    toc(text="Finished:")
    return rib_labels
    
    
###
def save_as_sparse(mat, fname):
    #spmat = sparse.coo_matrix(mat)  # (scipy.sparse)
    #sparse.save_npz(fname, spmat, compressed=True)
    spmat = sparse.COO.from_numpy(mat)
    pkl.dump(spmat, open(fname, "wb"))
    return
    
    
###           
    ## plot in 3d
    #plt.close('all')
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ##pos = np.where(binary_image==1)
    ##ax.scatter(pos[0], pos[1], pos[2], c='black')
    #color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(labels)))))
    
    #print("Number of volumes in image: ", len(unique_labels))
    ##print(vals)
    #for label in unique_labels:
    #    col = next(color)
    #    col[3] = 0.5
    #    #print(c)
    #    pos = np.where(labels==label)
    #    ax.text(pos[0][0], pos[1][0], pos[2][0], "---"+str(label), color='black', ha='left', va='center', zorder=1)#, bbox=dict(facecolor=’yellow’)) 
    #    ax.scatter(pos[0], pos[1], pos[2], color=col, zorder=3)
    #    #c = int(len(pos[0])/2)
        
    ##print("Plotted:")
    ##toc()
    #plt.show()






#############################################################################################################

#dicom_path = "/eresearch/lung/mpag253/Archive"
root = "/hpc/mpag253/Ribs/segmentation"
path_nifti = "/hpc/mpag253/Torso/segmentation"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Torso/torso_checklist.xlsx", skiprows=0, usecols=range(5)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_ribs_segmentation(input_list[i, 1:5], root, path_nifti, save_results=True,
                               troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")







