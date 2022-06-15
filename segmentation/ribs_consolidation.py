import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd


def run_ribs_consolidation(input_info, root, path_nifti, save_masks=True, troubleshooting_images=[False, False]):
    # Main script to ...
    
    [cohort, subject, condition, path_dicom] = input_info

    # Load the rib labels matrix
    rib_labels = load_sparse('Rib_Labels/rib_labels_'+subject+'_'+condition+'.pkl')
    
    
    # Fetch the allocation list for the subject
    allocation_list = retrieve_allocation_list(subject)
    
    # Consolidate labels
    ribs_consolidated = consolidate_rib_labels(rib_labels, allocation_list)
    
    # Save data
    save_as_sparse(ribs_consolidated, 'Rib_Labels/FINAL_rib_labels_'+subject+'_'+condition+'.pkl')
    
    # Plot to check consolidation
    plot_labelled_image(ribs_consolidated)
    
    # Get end points of each rib
    #rib_points = get_rib_points(ribs_consolidated)
    
    ###
    ###
    # REMEMBER TO PARAMETERISE THE DILATION/EROSION VALUES !!!
    # DO THE FLOATING RUBS GET TOO FAR AWAY FROM THE LUNGS?
    ###
    ###
    
    
###
def retrieve_allocation_list(subject):
    """This function neatly stores the allocations lists in a dictionary which can be retrieved.
    """
    dct = {}
    
    # Define the allocation lists
    # specifies which bodies in the above plot are allocated to which rib
    # e.g.    allocation_array = [[Right 1-12], [left 1-12], [Sternum]]
    #                          = [[[R1], [R2], ... [R12]],
    #                             [[L1], [L2], ... [L12]],
    #                             [[Sternum]]]
    dct['AGING003'] = [[[24], [13], [14], [15], [16], [17], [21], [22], [18, 25], [19], [23], [20]],
                       [[11], [9], [7], [6], [5], [4], [3], [1], [], [2], [8], [10], [11]],
                       [[12]]]
                       
    return dct[subject]
    
    
###
def plot_labelled_image(im):

    unique_labels = np.unique(im)[1:]
    print("Plot labels:\n", unique_labels)

    # plot in 3d
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #pos = np.where(binary_image==1)
    #ax.scatter(pos[0], pos[1], pos[2], c='black')
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(im))-1)))
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
def consolidate_rib_labels(rib_labels, allocation_list):
    
    ribs_consolidated = np.zeros(np.shape(rib_labels))
    for i in range(2):
        for j, rib in enumerate(allocation_list[i]):
            for rib_body in rib:
                ribs_consolidated[rib_labels==rib_body] = (i*12)+j+1
                
    return ribs_consolidated
    

###
def load_sparse(fname):
    spmat = pkl.load(open(fname, 'rb'))
    mat = sparse.COO.todense(spmat) 
    return mat  
    
    
###
def save_as_sparse(mat, fname):
    #spmat = sparse.coo_matrix(mat)  # (scipy.sparse)
    #sparse.save_npz(fname, spmat, compressed=True)
    spmat = sparse.COO.from_numpy(mat)
    pkl.dump(spmat, open(fname, "wb"))
    return

###
#def rib_points(im):
#    
#    for i in range(1,25):
        
    
    
    
    
    
    

#AGING001: [[27], [16,28],   [17], [18,30], [19], [20], [24], [25], [21,29], [22], [26], [23], ...
#           [12], [10,14], [8,13],     [7],  [6],  [5],  [4],  [2],     [1],  [3],  [9], [11], ...
#           [15]] 
    
    
    
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
        run_ribs_allocation(input_list[i, 1:5], root, path_nifti, save_masks=False,
                               troubleshooting_images=[True, False])
        # check_masks(cohort, subject, condition, root)
print("\n")


