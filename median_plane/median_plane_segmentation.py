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

            # Generate plane and save
            reg_c, reg_i = generate_median_image(image, image_info, threshold_lung, threshold_cort, gen_images=troubleshooting_images)        
            # Save plane coefficients to file
            if save_results:
                # Save the plane coefficients
                outfile = "output/median_plane_"+subject+"_"+condition
                save_plane_coefficients(outfile, reg_c, reg_i)

            print("\tRunning median segmentation...\t", subject, "\tDone.")


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
def generate_median_regions(lung_image, lungs_centre, lungs_span):
    """ Generates a binary image that specifies the valid region for ribs in the 3D image.
        Specifically, this method uses proximity to the lungs to define the rib region. 
    """
       
    # Inputs
    size_dilation = int(40/624*lungs_span[0]) #??/624 - optimised + generalised from AGING001
    size_erosion = int(20/624*lungs_span[0]) #??/624 - optimised + generalised from AGING001   

    # Prep
    # dilated lung region
    dilated_lungs = ndimage.binary_dilation(lung_image, iterations=size_dilation).astype(lung_image.dtype)
    # approximate block region for median
    block_width = int(80/624*lungs_span[0])
    mid_block = np.zeros(np.shape(lung_image)[1:])
    for i in range(len(mid_block)):
        if (i > lungs_centre[0]-block_width) and (i < lungs_centre[0]+block_width): 
            mid_block[i, :] = 1
    # lung convex hull
    cvlung = np.zeros(np.shape(lung_image))
    for i, axial_slice in enumerate(lung_image):
        cvlung[i, :, :] = skimage.morphology.convex_hull_image(lung_image[i, :, :])
    eroded_cvlung = ndimage.binary_erosion(cvlung, iterations=size_erosion).astype(lung_image.dtype)
    # anterior block
    ant_block = np.zeros(np.shape(lung_image)[1:])
    for j in range(np.shape(ant_block)[1]):
        if j > (lungs_centre[1]): 
            ant_block[:, j] = 1
    
    # Generate spine region
    spine_region = np.multiply(1-dilated_lungs, mid_block)

    # Generate sternum region
    strnm_region = np.multiply(np.multiply(1-eroded_cvlung, mid_block), ant_block)
    

    return spine_region, strnm_region
      

###
def segment_spine_median(image, spine_region, threshold=600, gen_images=[False, False]):
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
    #generate_img_fig(rib_region[smpl_slc,:,:], case_name+"2P", "rib region", gen_im1)
    binary_image = np.array(image > threshold, dtype=np.int8) + 0  # bg=0, bone=1
    #generate_img_fig(binary_image[smpl_slc,:,:], case_name+"3P", "thresholded", gen_im1)  

    #Mask the image
    binary_image = np.multiply(binary_image, spine_region)

    ## Keep only the lowest y-indices
    # Take mean of remaining pixels
    for i, axial_slice in enumerate(binary_image):
    
        # keep lowest
        indices = np.nonzero(axial_slice)
        min_indices = np.argsort(indices[1])[:50]  # X lowest
        binary_image[i,:,:] = 0
    
        # take mean
        if len(min_indices) > 0:
            x_mean = np.mean(indices[0][min_indices]).astype(int)
            y_mean = np.mean(indices[1][min_indices]).astype(int)
            binary_image[i, x_mean, y_mean] = 1
            
    # To generate images for all slices...
    if False:
        for i in range(len(binary_image)):
            if i%20 == 0:
                generate_img_fig(binary_image[i, :, :], "images/chest_slice_"+"{:03d}".format(i), "slice image", gen_im1)
                #+median_region[:, :, i]
                plt.close('all')

    ## Regenerate final rib centroids
    #for i, axial_slice in enumerate(eroded_ribs):
    #    #axial_slice = axial_slice - 1 # converts to pseudo-binary: bone=1, rest=0
    #    labels = skimage.measure.label(axial_slice) #, connectivity=1)
    #    centroids, label_list = centroids_of_labels(labels, bg=0)
    #    binary_image[i] = 0
    #    for centroid in centroids[1:]:
    #        binary_image[i][centroid[0]][centroid[1]] = 1
            
    return binary_image
    
    
###
def segment_strnm_median(image, strnm_region, threshold=600, gen_images=[False, False]):
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
    #generate_img_fig(rib_region[smpl_slc,:,:], case_name+"2P", "rib region", gen_im1)
    binary_image = np.array(image > threshold, dtype=np.int8) + 0  # bg=0, bone=1
    #generate_img_fig(binary_image[smpl_slc,:,:], case_name+"3P", "thresholded", gen_im1)  

    #Mask the image
    binary_image = np.multiply(binary_image, strnm_region)

    # Keep only the large solid structures (in each slice)
    for i, axial_slice in enumerate(binary_image):
        labels = skimage.measure.label(axial_slice) #, connectivity=1)
        l_vals, l_counts = size_of_labels(labels, bg=0)
        binary_image[i, :, :] = 0
        for j in range(len(l_vals)):
            if l_counts[j] > 50:    
               binary_image[i][labels == l_vals[j]] = 1  # soft/air=1, bone=2, large_bone=3
   
    # Take mean of remaining pixels
    for i, axial_slice in enumerate(binary_image):
        indices = np.nonzero(axial_slice)
        binary_image[i, :, :] = 0
        if len(indices[0]) > 0:
            x_mean = np.mean(indices[0]).astype(int)
            y_mean = np.mean(indices[1]).astype(int)
            binary_image[i, x_mean, y_mean] = 1
        
    # To generate images for all slices...
    if False:
        for i in range(len(binary_image)):
            if i%20 == 0:
                generate_img_fig(binary_image[i, :, :]+strnm_region[i,:,:], "images/chest_slice_"+"{:03d}".format(i), "slice image", gen_im1)
                #+strnm_region[i,:,:]
                plt.close('all')

    ## Regenerate final rib centroids
    #for i, axial_slice in enumerate(eroded_ribs):
    #    #axial_slice = axial_slice - 1 # converts to pseudo-binary: bone=1, rest=0
    #    labels = skimage.measure.label(axial_slice) #, connectivity=1)
    #    centroids, label_list = centroids_of_labels(labels, bg=0)
    #    binary_image[i] = 0
    #    for centroid in centroids[1:]:
    #        binary_image[i][centroid[0]][centroid[1]] = 1
            
    return binary_image

###
def save_as_sparse(mat, fname):
    #spmat = sparse.coo_matrix(mat)  # (scipy.sparse)
    #sparse.save_npz(fname, spmat, compressed=True)
    spmat = sparse.COO.from_numpy(mat)
    pkl.dump(spmat, open(fname, "wb"))
    return

    
###
def get_label_coordinates(rib_labels, image_info):
    """
    """
    
    # unpack image info
    [d1, d2, d3, p1, p2, p3] = image_info
    
    # find the array indices of labelled pixels
    indices = np.nonzero(rib_labels)
    
    # number of labels
    n_labels = len(indices[0])
    
    # pre-allocate output array
    label_coordinates = np.empty([n_labels, 5])
    
    # generate the coordinates for each label in the image    
    for i in range(n_labels):
        rib_label = rib_labels[indices[0][i], indices[1][i], indices[2][i]]
        x_coord = p1/2 + p1*indices[0][i]
        y_coord = -(p2/2 + p2*indices[1][i]) + d2*p2
        z_coord = p3/2 + p3*indices[2][i] - d3*p3
        label_coordinates[i, :] = [rib_label, i+10001, x_coord, y_coord, z_coord]
        
    return label_coordinates
    

###
def fit_plane_to_coordinates(label_coords):

    [x, y, z] = label_coords

    ## your data is stored as X, Y, Z
    #print(X.shape, Y.shape, Z.shape)

    #x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()

    #X_data = np.array([x1, y1]).reshape((-1, 2))
    #Y_data = z1
    
    xy = np.vstack((x,y)).T
    from sklearn import linear_model
    reg = linear_model.LinearRegression().fit(xy, z)
    #print("coefficients of equation of plane, (a1, a2): ", reg.coef_)
    #print("value of intercept, c:", reg.intercept_)
    
    return reg.coef_, reg.intercept_
    
    
###
def save_plane_coefficients(fname, reg_c, reg_i):
    
    #lines = [str(reg_c[0])+"\n", str(reg_c[1])+"\n", str(reg_i)]
    #f = open(fname+".txt", "a")
    #f.writelines(lines)
    #f.close()
    np.save(fname, [reg_c[0], reg_c[1], reg_i])
    return
    
    
###    
def plot_im(im):       
    
    # set up plot
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    pos = np.where(im>0)
    ax.scatter(pos[0], pos[1], pos[2], c='black')
    
    #color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(im)))))
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
    plt.show()

 
###   
def generate_median_image(image, image_info, threshold_lung, threshold_cort, gen_images=[False, False]):
  
    tic()
    
    # Generate lung segmentation and extract measures of the lung
    lung_image = segment_lungs(image, threshold=threshold_lung, gen_images=[False, False])     
    # generate measures from the lung image
    lungs_centre, lungs_span = generate_lung_measures(lung_image)
    #toc(text="Lung image + measures:")
            
    # Generate the median region image
    lung_image = np.moveaxis(lung_image, 2, 0)     # switch axes to enumerate axial slices easily
    spine_region, strnm_region = generate_median_regions(lung_image, lungs_centre, lungs_span)
    #spine_region = np.moveaxis(spine_region, 0, 2) # revert axes to original
    #toc(text="Rib region:")
    
    # Generate image of median-defining bodies
    # (sternum, spine ... TBD)
    image = np.moveaxis(image, 2, 0) # switch axes to enumerate easily (ORIGINAL -> AXIAL)
    spine_median = segment_spine_median(image, spine_region, threshold=1200, gen_images=gen_images)
    strnm_median = segment_strnm_median(image, strnm_region, threshold=600, gen_images=gen_images)
    
    both_medians = spine_median + strnm_median
    plot_im(both_medians)
    
    
    # To generate images for all slices...
    if False:
        for i in range(len(both_medians)):
            if i%20 == 0:
                generate_img_fig(both_medians[i, :, :], "images/chest_slice_"+"{:03d}".format(i), "slice image", gen_images[0])
                #+median_region[:, :, i]
                plt.close('all')
                
    # Get the coordinates for each label in the image
    label_coords = get_label_coordinates(both_medians, image_info)
    
    # Fit plane to the coordinates
    reg_c, reg_i = fit_plane_to_coordinates(np.moveaxis(label_coords, -1, 0)[2:])
        
    return reg_c, reg_i
    





#############################################################################################################

#dicom_path = "/eresearch/lung/mpag253/Archive"
root = "/hpc/mpag253/Ribs/median_plane"
path_nifti = "/hpc/mpag253/Torso/segmentation"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Torso/torso_checklist.xlsx", skiprows=0, usecols=range(11)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_median_segmentation(input_list[i, 1:5], input_list[i, 5:], root, path_nifti, save_results=True,
                               troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")







