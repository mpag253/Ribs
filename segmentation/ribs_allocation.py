import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd


def run_ribs_allocation(input_info, root, path_nifti, save_masks=True, troubleshooting_images=[False, False]):
    """ Main script to ...
    """
    
    [cohort, subject, condition, path_dicom] = input_info

    # Load the rib labels matrix
    ribs_image = load_sparse('Rib_Labels/ribs_segmented_'+subject+'_'+condition+'.pkl')
    
    # Plot for allocation
    print("\tPlotting... "+subject, end="\r")
    plot_labelled_image(ribs_image)
    print("\tPlotting... "+subject+"\tDone.")
    

def size_of_labels(im, bg=-1):
    """ Returns the number of voxels for each label in a labelled image, 'im'.
        Excludes the background specified by 'bg'.
    """
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    return vals, counts
        
    
###
def plot_labelled_image(im):

    unique_labels = np.unique(im)[1:]
    #print("\t\t(number of labels: "+str(len(unique_labels))+")")

    # plot in 3d
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #pos = np.where(binary_image==1)
    #ax.scatter(pos[0], pos[1], pos[2], c='black')
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))
    #print("Number of volumes in image: ", len(unique_labels))
    #print(vals)
    for label in unique_labels:
        col = next(color)
        col[3] = 0.5
        #print(c)
        pos = np.where(im==label)
        ax.text(pos[0][0], pos[1][0], pos[2][0], "---"+str(int(label)), color='black', ha='left', va='center', zorder=1)#, bbox=dict(facecolor=’yellow’)) 
        ax.scatter(pos[0], pos[1], pos[2], color=col, zorder=3)
        #c = int(len(pos[0])/2)
    plt.show()
      

###
def load_sparse(fname):
    spmat = pkl.load(open(fname, 'rb'))
    mat = sparse.COO.todense(spmat) 
    return mat  
        
    
#############################################################################################################

#dicom_path = "/eresearch/lung/mpag253/Archive"
root = "/hpc/mpag253/Ribs/segmentation"
path_nifti = "/hpc/mpag253/Torso/segmentation"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/ribs_checklist.xlsx", skiprows=0, usecols=range(5)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_ribs_allocation(input_list[i, 1:5], root, path_nifti, save_masks=False,
                               troubleshooting_images=[True, False])
        # check_masks(cohort, subject, condition, root)
print("\n")


