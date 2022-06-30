#import natsort
#import os
#import nibabel as nib
import numpy as np
#import dicom2nifti
import matplotlib.pyplot as plt
#from PIL import Image
#import skimage
#import skimage.measure
#from scipy import ndimage
import sparse
import pandas as pd
import pickle as pkl


def plot_median_plane(input_info, image_info, root, path_ribs):
    """
    """
    
    # Setup
    [cohort, subject, condition, path_dicom] = input_info
    [d1, d2, d3, p1, p2, p3] = image_info
    print("\tPlotting median plane...\t", subject, end="\r")
    
    # Define image bounds
    bounds = [0, d1*p1,   0, d2*p2,   -d3*p3, 0]  # [x0, x1, y0, y1, z0, z1]
    
    # Load the ribs data
    #im = load_sparse(path_ribs+'/Rib_Labels/ribs_segmented_'+subject+'_'+condition+'.pkl')
    data = load_ipdata(path_ribs+'/output/'+cohort+'/'+subject+'/'+condition+'/Ribs/Data/ribs_data_ALL')
    
    # Load the plane
    plane = np.load(root+"/output/median_plane_"+subject+"_"+condition+".npy")
    print(plane)
    
    # Plot
    plot_data_and_plane(data, plane, bounds)
    

###
def load_ipdata(fname):
    f = open(fname+".ipdata", "r")
    lines = f.readlines()
    data = np.empty([len(lines)-1, 3])
    for i, line in enumerate(lines[1:]):
        data[i, :] = [float(j) for j in line.split(" ")[2:5]]
    return data



###
def load_sparse(fname):
    spmat = pkl.load(open(fname, 'rb'))
    mat = sparse.COO.todense(spmat) 
    return mat  


###
def plot_data_and_plane(data, plane, bounds):

    # set up 3d plot
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ## No rib labels
    #pos = np.where(im>0)
    #ax.scatter(pos[0], pos[1], pos[2], c='black')
    # Rib labels
    #unique_labels = np.unique(im)[1:]
    ##print("Plot labels: ", unique_labels)
    #colours = plt.cm.rainbow(np.linspace(0, 1, 24)) 
    ##print("Number of volumes in image: ", len(unique_labels))
    ##print(vals)
    #for label in unique_labels: 
    #    col = colours[ int((label-1) - (label%2-1)*11 - (label>12)*11) ] # just splitting up colours for contrast
    #    col[3] = 0.5
    #    #print(c)
    #    pos = np.where(im==label)
    #    ax.text(pos[0][0], pos[1][0], pos[2][0], "---"+str(int(label)), color='black', ha='left', va='center', zorder=1)#, bbox=dict(facecolor=’yellow’)) 
    #    ax.scatter(pos[0], pos[1], pos[2], color=col, zorder=3)
    #    #c = int(len(pos[0])/2)
    
    # Plot the data
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='black')
        
    # Plot the plane
    
    x = bounds[0:2]
    y = bounds[2:4]
    xx, yy = np.meshgrid(x, y)
    zz=plane[0]*xx + plane[1]*yy + plane[2]
    
    
    #y = bounds[2:4]
    #z = bounds[4:6]
    #yy, zz = np.meshgrid(y, z)
    #xx = (zz - plane[1]*yy - plane[2])/plane[0]
    
    ax.plot_surface(xx, yy, zz, alpha=0.2)
        
        
    plt.show()










#############################################################################################################

#dicom_path = "/eresearch/lung/mpag253/Archive"
root = "/hpc/mpag253/Ribs/median_plane"
path_ribs = "/hpc/mpag253/Ribs/point_clouds"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/ribs_checklist.xlsx", skiprows=0, usecols=range(11)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        plot_median_plane(input_list[i, 1:5], input_list[i, 5:], root, path_ribs)
        # check_masks(cohort, subject, condition, root)
print("\n")







