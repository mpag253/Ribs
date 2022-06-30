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


def run_median_segmentation(input_info, image_info, plane_points, root, rib_label_dir, save_results=True, troubleshooting_images=[False, False]):
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
    print("\tRunning median fitting...\t", subject, end="\r")

    # Load the labelled ribs image
    im = load_sparse(rib_label_dir+'/ribs_labelled_'+subject+'_'+condition+'.pkl')
    
    # Fit plane
    plane = fit_median_plane(plane_points, image_info, im)
    
    # Save plane coefficients to file
    if save_results:
        # Save the plane coefficients
        export_dir = root+"/output/"+cohort+"/"+subject+"/"+condition+"/Median_Plane/"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        outfile = export_dir+"median_plane"
        save_plane_coefficients(outfile, plane)
        
    # Plot
    plt.show()

    print("\tRunning median fitting...\t", subject, "\tDone.")

   
###
def get_coordinates_of_indices(indices, image_info):
    """
    """
    
    # unpack image info
    [d1, d2, d3, p1, p2, p3] = image_info
    
    # generate the coordinates
    y_i, x_i, z_i = indices  # xyz out of order becuase of selection
    x_coord = p1/2 + p1*x_i
    y_coord = -(p2/2 + p2*y_i) + d2*p2
    z_coord = p3/2 + p3*z_i - d3*p3
        
    return [x_coord, y_coord, z_coord]
    

###
def fit_plane_to_coordinates(coords):

    [x, y, z] = coords.T

    #xy = np.vstack((x,y)).T
    yz = np.vstack((y,z)).T
    from sklearn import linear_model
    reg = linear_model.LinearRegression().fit(yz, x)
    #print("coefficients of equation of plane, (a1, a2): ", reg.coef_)
    #print("value of intercept, c:", reg.intercept_)
    
    plane_abcd = [1, -reg.coef_[0], -reg.coef_[1], -reg.intercept_]
    
    return plane_abcd
    
    
###
def save_plane_coefficients(fname, plane):
    np.save(fname, plane)
    return
    
    
###
def load_sparse(fname):
    f = open(fname, 'rb')
    spmat = pkl.load(f)
    f.close()
    mat = sparse.COO.todense(spmat) 
    return mat  
    
    
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
def plot_im(plane, coords, label_coords):       
    
    # set up plot
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # plot points
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='black')
    
    # plot plane
    y = [np.min(coords[:, 1]), np.max(coords[:, 1])]
    z = [np.min(coords[:, 2]), np.max(coords[:, 2])]
    yy, zz = np.meshgrid(y, z)
    xx = (-plane[1]*yy - plane[2]*zz - plane[3])/plane[0]
    ax.plot_surface(xx, yy, zz, alpha=0.2)
    
    # plot rib labels
    # label_coords --> [rib_label, i+10001, x_coord, y_coord, z_coord]
    unique_labels = np.unique(label_coords[:, 0])
    colours = plt.cm.rainbow(np.linspace(0, 1, 24)) 
    for label in unique_labels: 
        col = colours[ int((label-1) - (label%2-1)*11 - (label>12)*11) ] # just splitting up colours for contrast
        col[3] = 0.5
        label_indices = label_coords[:, 0]==label
        label_pts = label_coords[label_indices, 2:5]
        ax.scatter(label_pts[:, 0], label_pts[:, 1], label_pts[:, 2], color=col, zorder=3)
    
    return


 
###   
def fit_median_plane(points, image_info, im):

    #remove nan data
    points = points[~pd.isnull(points)]
    
    # number of points
    n_pts = int(len(points)/3)
    
    # preallocate coordinates array
    coords = np.empty([n_pts, 3])
                
    # Get the coordinates for each label in the image
    for i in range(n_pts):
        indices = points[(3*i):(3*i+3)]
        coords[i, :] = get_coordinates_of_indices(indices, image_info)
    
    # Fit plane to the coordinates
    plane = fit_plane_to_coordinates(coords)
    
    # Convert labelled im to coordinates for plotting
    label_coords = get_label_coordinates(im, image_info)
    
    # Generate plot
    plot_im(plane, coords, label_coords)    
        
    return plane
    





#############################################################################################################

#dicom_path = "/eresearch/lung/mpag253/Archive"
root = "/hpc/mpag253/Ribs/median_plane"
rib_label_dir = "/hpc/mpag253/Ribs/segmentation/Rib_Labels"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/median_plane/points_for_median_plane.xlsx", skiprows=0, usecols=range(36)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_median_segmentation(input_list[i, 1:5], input_list[i, 5:11], input_list[i, 12:36], root, rib_label_dir, save_results=True,
                               troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")







