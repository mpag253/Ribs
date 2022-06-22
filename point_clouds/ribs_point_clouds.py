import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd


def run_ribs_consolidation(input_info, root, save_masks=True, troubleshooting_images=[False, False]):
    """ Main script to ...
    """
    
    # unpack
    [cohort, subject, condition, path_dicom, d1, d2, d3, p1, p2, p3] = input_info
    print("\tGenerating point clouds... "+subject+" "+condition, end="\r")

    # Load the rib labels matrix
    rib_label_dir = '/hpc/mpag253/Ribs/segmentation/Rib_Labels/'
    rib_labels = load_sparse(rib_label_dir+'ribs_labelled_'+subject+'_'+condition+'.pkl')
    
    # Get the coordinates for each label in the image
    label_coordinates = get_label_coordinates(rib_labels, p1, p2, p3)
    
    # Write the coordinates to ipdata file
    export_fname = root+"/Ribs_Point_Clouds/ribs_data_"+subject+"_"+condition
    ipdata_header = 'ribs centre points: '+subject+" "+condition
    exdata_group_name = "centreline_Ribs"
    export_to_ipdata(label_coordinates[:, 1:], export_fname, ipdata_header)
    export_to_exdata(label_coordinates[:, 1:], export_fname, exdata_group_name)
    
    print("\tGenerating point clouds... "+subject+" "+condition+"\t Done.")
    
    return


###
def export_to_exdata(dat, filename, group_name):
    
    # generate the lines file
    n_points = len(dat)
    exdata_lines = [[] for _ in range(n_points+6)]
    exdata_lines[0] = " Group name: "+group_name+"\n"
    exdata_lines[1] = " #Fields=1\n"
    exdata_lines[2] = " 1) coordinates, coordinate, rectangular cartesian, #Components=3\n"
    exdata_lines[3] = "  x.  Value index=1, #Derivatives=0, #Versions=1\n"
    exdata_lines[4] = "  y.  Value index=2, #Derivatives=0, #Versions=1\n"
    exdata_lines[5] = "  z.  Value index=3, #Derivatives=0, #Versions=1\n"
    for i in range(n_points):
        exdata_lines[i+6] = " Node: {:d}\n {:13.6e}\n {:13.6e}\n {:13.6e}\n".format(int(dat[i,0]), dat[i,1], dat[i,2], dat[i,3])
    
    # export
    file_out = open(filename+".exdata", 'w')
    file_out.writelines(exdata_lines)
    file_out.close()


###
def export_to_ipdata(dat, filename, header):
    
    # generate the lines file
    n_points = len(dat)
    ipdata_lines = [[] for _ in range(n_points+1)]
    ipdata_lines[0] = header+"\n"
    for i in range(n_points):
        ipdata_lines[i+1] = " {:>5d} {:12.6e} {:12.6e} {:12.6e} 1.0 1.0 1.0\n".format(int(dat[i,0]), dat[i,1], dat[i,2], dat[i,3])
    
    # export
    file_out = open(filename+".ipdata", 'w')
    file_out.writelines(ipdata_lines)
    file_out.close()


###
def get_label_coordinates(rib_labels, p1, p2, p3):
    """
    """
    
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
        y_coord = p2/2 + p2*indices[1][i]
        z_coord = p3/2 + p3*indices[2][i]
        label_coordinates[i, :] = [rib_label, i+1, x_coord, y_coord, z_coord]
        
    return label_coordinates
    

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

    
    
#############################################################################################################

root = "/hpc/mpag253/Ribs/point_clouds"
#paths = [dicom_path, root, ]
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/ribs_checklist.xlsx", skiprows=0, usecols=range(11)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_ribs_consolidation(input_list[i, 1:11], root, save_masks=False,
                               troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")


