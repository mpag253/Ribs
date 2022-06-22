import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd


def run_ribs_consolidation(input_info, allocation_dict, root, path_nifti, save_masks=True, troubleshooting_images=[False, False]):
    """ Main script to ...
    """
    
    [cohort, subject, condition, path_dicom] = input_info

    # Load the rib labels matrix
    rib_labels = load_sparse('Rib_Labels/ribs_segmented_'+subject+'_'+condition+'.pkl')
    
    # Fetch the allocation list for the subject
    allocation_list = allocation_dict[subject]
    
    # Consolidate labels
    ribs_consolidated = consolidate_rib_labels(rib_labels, allocation_list)
    
    # Save data
    save_as_sparse(ribs_consolidated, 'Rib_Labels/ribs_labelled_'+subject+'_'+condition+'.pkl')
    
    # Plot to check consolidation
    print("Plotting... "+subject)
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
def load_allocation_dict():
    """This function neatly stores the allocations lists in a dictionary which can be retrieved.
    """
    dct = {}
    
    data = input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/segmentation/ribs_data_for_allocation.xlsx", skiprows=0, usecols=range(15)))
    print(data,"\n")
    
    # Define the allocation lists
    # specifies which bodies in the above plot are allocated to which rib
    # e.g.    allocation_array = [[Right 1-12], [left 1-12], [Sternum]]
    #                          = [[[R1], [R2], ... [R12]],
    #                             [[L1], [L2], ... [L12]],
    #                             [[Sternum]]]
    
    for i, line in enumerate(data):
        if line[1] == 'Right':
            subject = line[0]
            dict_rght = [[] for _ in range(12)]
            dict_left = [[] for _ in range(12)]
            dict_sternum = [[]]
            for j in range(12):
                
                # Right ribs
                entry_rght = data[i+0][j+3]
                if isinstance(entry_rght, int):
                    dict_rght[j] = [entry_rght]
                elif isinstance(entry_rght, str):
                    dict_rght[j] = [int(entry) for entry in str(entry_rght).split(",")]
                
                # Left ribs
                entry_left = data[i+1][j+3]
                if isinstance(entry_left, int):
                    dict_left[j] = [entry_left]
                elif isinstance(entry_left, str):
                    dict_left[j] = [int(entry) for entry in str(entry_left).split(",")]
            
            # Sternum
            entry_sternum = data[i+2][3]
            if isinstance(entry_sternum, int):
                dict_sternum[0] = [entry_sternum]
            elif isinstance(entry_sternum, str):
                dict_sternum[0] = [int(entry) for entry in str(entry_sternum).split(",")]
            
            dct[subject] = [dict_rght, dict_left, dict_sternum]
            #print(dct[subject])

    
                       
    return dct
    
    
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
    dct['AGING001'] = [[    [28], [17], [18], [21], [20], [19], [23,30], [24], [26,29], [27], [25], [22]],
                       [[12, 13], [10],  [8],  [7],  [6],  [5],     [4],  [2],     [1],  [3],  [9], [11]],
                       [[15]]]
    
    dct['AGING002'] = [[      [31], [33], [32,34], [29,30], [28], [26], [27],   [23], [24], [21,25], [20],   []],
                       [[11,12,13], [10],     [8],  [7,18],  [6],  [5],  [3], [2,15],  [1],     [4],  [9], [14]],
                       [[19]]]
    
    dct['AGING003'] = [[[41], [37,42], [27], [26,35], [29], [31], [30], [32,39], [33], [34], [36], [38]],
                       [[15], [9], [8,19], [7,12,14], [5,13], [4], [3], [20,2], [1], [6], [10], [16]],
                       [[24]]]
    
    dct['AGING004'] = [[[32], [33,27], [26], [25,34], [24], [20,30], [23], [22,31], [19], [18,29], [21], [28]],
                       [[11], [10,13], [8], [7], [6], [5], [4], [2], [1,15], [3,14], [9], [12]],
                       [[17]]]
    
    dct['AGING006'] = [[[27,35], [34], [26], [31,36], [30], [23], [25,38], [24], [29], [28], [33], [32]],
                       [[14,15], [11], [9], [7], [6,13,16], [5], [3,10], [2], [1], [4], [8], [12,17]],
                       [[21,22]]]
                       
    return dct[subject]
    
    
###
def plot_labelled_image(im):

    unique_labels = np.unique(im)[1:]
    print("Plot labels: ", unique_labels)

    # plot in 3d
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #pos = np.where(binary_image==1)
    #ax.scatter(pos[0], pos[1], pos[2], c='black')
    colours = plt.cm.rainbow(np.linspace(0, 1, 24)) 
    #print("Number of volumes in image: ", len(unique_labels))
    #print(vals)
    for label in unique_labels: 
        col = colours[ int((label-1) - (label%2-1)*11 - (label>12)*11) ] # just splitting up colours for contrast
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
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/ribs_checklist.xlsx", skiprows=0, usecols=range(5)))
allocation_dict = load_allocation_dict()

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_ribs_consolidation(input_list[i, 1:5], allocation_dict, root, path_nifti, save_masks=False,
                               troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")


