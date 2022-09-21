import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd
import os


def run_ribs_point_clouds(input_info, image_info, root, save_masks=True, troubleshooting_images=[False, False]):
    """ Main script to ...
    """
    
    # unpack
    [cohort, subject, condition, path_dicom] = input_info
    #[d1, d2, d3, p1, p2, p3] = image_info
    print("\tGenerating point clouds... "+subject+" "+condition, end="\r")

    # Load the rib labels matrix
    rib_label_dir = '/hpc/mpag253/Ribs/segmentation/Rib_Labels/'
    rib_labels = load_sparse(rib_label_dir+'ribs_labelled_'+subject+'_'+condition+'.pkl')
    
    # Get the coordinates for each label in the image
    label_coordinates = get_label_coordinates(rib_labels, image_info)
    
    # Define IDs for each rib label
    rib_ids = ["ALL", "T01R", "T02R", "T03R", "T04R", "T05R", "T06R", "T07R", "T08R", "T09R", "T10R", "T11R", "T12R",
                      "T01L", "T02L", "T03L", "T04L", "T05L", "T06L", "T07L", "T08L", "T09L", "T10L", "T11L", "T12L"] 
    
    ## Write the coordinates to file - all ribs + each rib individually
    export_dir = root+"/output/"+cohort+"/"+subject+"/"+condition+"/Ribs/Data/"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    for i in range(len(rib_ids)):
        rib_id = rib_ids[i]
        export_fname = export_dir+"ribs_data_"+rib_id
        ipdata_header = 'ribs centre points: '+subject+" "+condition+" "+rib_id
        exdata_group_name = "Rib_"+rib_id
        if i==0:
            export_to_ipdata(label_coordinates[:, 1:], export_fname, ipdata_header)
            export_to_exdata(label_coordinates[:, 1:], export_fname, exdata_group_name) 
        else:
            label_indices = label_coordinates[:, 0]==i
            export_to_ipdata(label_coordinates[label_indices, 1:], export_fname, ipdata_header)
            export_to_exdata(label_coordinates[label_indices, 1:], export_fname, exdata_group_name)    
    
    print("\tGenerating point clouds... "+subject+" "+condition+"\t Done.")
    
    return


###
def get_label_coordinates(rib_labels, image_info):
    """
    """
    
    #unpack image info
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
        run_ribs_point_clouds(input_list[i, 1:5], input_list[i, 5:], root, save_masks=False,
                              troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")


