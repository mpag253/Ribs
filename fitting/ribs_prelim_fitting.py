import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd
from scipy import interpolate
import os
import copy
from sys import exit


def run_ribs_prelim_fitting(input_info, image_info, root, save_masks=True, troubleshooting_images=[False, False]):
    """ Main script to ...
    """
    
    # unpack
    [cohort, subject, condition, path_dicom] = input_info
    #[d1, d2, d3, p1, p2, p3] = image_info
    print("\tPreliminary fitting... "+subject+" "+condition, end="\r")
    
    # Load the rib point clouds - transformed in aligned reference frame
    point_cloud_dir = '/hpc/mpag253/Ribs/point_clouds/output/'+cohort+"/"+subject+"/"+condition+"/Ribs/Data/"
    median_dir = '/hpc/mpag253/Ribs/median_plane/output/'+cohort+"/"+subject+"/"+condition+'/Median_Plane/'
    # load the median plane
    median_plane = np.load(median_dir + "median_plane.npy")
    # calculate the reorientation vector
    normal_vector = median_plane[:3]
    unit_normal_vector = normal_vector/np.linalg.norm(median_plane[:3]) 
    # load the point clouds and transform
    labelled_coords = load_point_clouds(point_cloud_dir, unit_normal_vector)
    
    # Load the landmarks
    fname = torso_path+"landmarks/"+cohort+"/"+subject+"/"+condition+"/landmarks.txt"
    with open(fname, "r") as f:
        lines = f.readlines()
    landmarks = [float(i) for i in lines]

    # Move and scale
    fname = torso_path+"landmarks/Human_Aging/AGING001/EIsupine/landmarks.txt"
    with open(fname, "r") as f:
        lines = f.readlines()
    landmarks_reference = [float(i) for i in lines[:2]]
    height_t2t11 = landmarks[1] - landmarks[0]
    height_t2t11_ref = landmarks_reference[1] - landmarks_reference[0]
    scale_factor = height_t2t11/height_t2t11_ref
    coords_scaled = copy.deepcopy(labelled_coords)
    coords_scaled[:, 2:] /= scale_factor
    coords_scaled[:, 4] += landmarks_reference[1] - landmarks[1]/scale_factor
    
    # Trim
    trim_left = landmarks[2] + 35
    trim_rght = landmarks[2] - 35
    keep_indices = (coords_scaled[:, 2] > trim_left) + (coords_scaled[:, 2] < trim_rght)
    coords_trimmed = coords_scaled[keep_indices]
    #plot_labelled_coords(coords_trimmed)
    
    # Define centre
    centre_xy = [landmarks[2], np.max(coords_trimmed[:, 3]) - 100]
    
    # Convert to cylindrical coordinates
    coords_cyl = copy.deepcopy(coords_trimmed)
    #coords_cylindrical[:, 2:4] -= centre_xy
    #plot_labelled_coords(coords_cylindrical)
    xy_prime = coords_trimmed[:, 2:4] - centre_xy
    angles = np.arctan2(xy_prime[:, 0], xy_prime[:, 1])
    radii = np.sqrt(np.square(xy_prime[:, 0]) + np.square(xy_prime[:, 1]))
    coords_cyl[:, 2] = angles
    coords_cyl[:, 3] = radii
    
    # Resample for node points
    rib_angles = np.array([[0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0, 0, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40, 0, 0],
                           [2.30, 2.30, 2.30, 2.30, 2.30, 2.20, 2.00, 1.80, 1.60, 1.40, 0, 0, -2.30, -2.30, -2.30, -2.30, -2.30, -2.20, -2.00, -1.80, -1.60, -1.40, 0, 0]])
    rib_n_nodes = [20, 20, 20, 20, 20, 19, 17, 15, 13, 11, 0, 0, 20, 20, 20, 20, 20, 19, 17, 15, 13, 11, 0, 0]
    resampled_points_cyl = np.empty([0, 5])
    unique_labels = np.unique(coords_cyl[:, 0])
    for label in unique_labels:
        i = int(label)-1
        label_indices = coords_cyl[:, 0]==label
        label_coords = coords_cyl[label_indices, 2:]
        # raw angles
        #min_angle_i = np.argmin(label_coords[:, 0])
        #max_angle_i = np.argmax(label_coords[:, 0])
        #print(label, label_coords[min_angle_i, 0], label_coords[max_angle_i, 0])
        # resample
        if len(label_coords[:, 0]) > 1:
            f_r = interpolate.interp1d(label_coords[:, 0], label_coords[:, 1], bounds_error=False)
            f_z = interpolate.interp1d(label_coords[:, 0], label_coords[:, 2], bounds_error=False)
            label_thetas = np.linspace(rib_angles[0, i], rib_angles[1, i], num=rib_n_nodes[i])
            label_radii = f_r(label_thetas)
            label_z = f_z(label_thetas)
        else:
            label_thetas = []
            label_radii = []
            label_z = []
        # store
        for j in range(len(label_thetas)):
            new_row = [label, len(resampled_points_cyl)+1, label_thetas[j], label_radii[j], label_z[j]]
            resampled_points_cyl = np.vstack((resampled_points_cyl, new_row))
        
    # Convert back to cartesian coordinates
    resampled_points_cart = copy.deepcopy(resampled_points_cyl)
    resampled_x = np.multiply(resampled_points_cyl[:, 3], np.sin(resampled_points_cyl[:, 2]))
    resampled_y = np.multiply(resampled_points_cyl[:, 3], np.cos(resampled_points_cyl[:, 2]))
    resampled_points_cart[:, 2] = resampled_x
    resampled_points_cart[:, 3] = resampled_y
    #plot_labelled_coords(resampled_points_cart)
    
    # Save to ipnode
    fdir = '/hpc/mpag253/Ribs/fitting/output/'+cohort+"/"+subject+"/"+condition+"/Ribs/"
    os.makedirs(fdir, exist_ok=True)
    fname = fdir+"Ribs_fitted.ipnode"
    save_labelled_coords_to_ipnode(resampled_points_cart, fname)
    
    print("\tPreliminary fitting... "+subject+" "+condition+"\t Done.")
    return


###
def save_labelled_coords_to_ipnode(labelled_coords, fname_out):

    unique_labels = np.unique(labelled_coords[:, 0])
    
    fname_template = '/hpc/mpag253/Ribs/fitting/ipnode_template.ipnode'
    with open(fname_template, 'r') as f:
        lines = f.readlines()
    
    non_nan_indices = ~np.isnan(labelled_coords[:, 4])
    n_nodes = len(labelled_coords[non_nan_indices, 1])
    lines[3] = ' The number of nodes is [{:>5d}]: {:>5d}\n'.format(n_nodes, n_nodes)
    
    for i in range(len(labelled_coords[:, 1])):
        if ~np.isnan(labelled_coords[i, 4]):
            node_num = 100*int(labelled_coords[i, 0]) + int(labelled_coords[i, 1]) + 10000
            
            lines.append(' Node number [{:>5d}]: {:>5d}\n'.format(node_num, node_num))
            lines.append(' The number of versions for nj=1 is [1]: 1\n')
            lines.append(' The Xj(1) coordinate is [ 0.00000E+00]: {:>12.6e}\n'.format(labelled_coords[i, 2]))
            lines.append(' The number of versions for nj=2 is [1]: 1\n')
            lines.append(' The Xj(2) coordinate is [ 0.00000E+00]: {:>12.6e}\n'.format(labelled_coords[i, 3]))
            lines.append(' The number of versions for nj=3 is [1]: 1\n')
            lines.append(' The Xj(3) coordinate is [ 0.00000E+00]: {:>12.6e}\n'.format(labelled_coords[i, 4]))
            lines.append('\n')
        
    with open(fname_out, 'w') as f:
        f.writelines(lines)
        

###
def load_point_clouds(point_cloud_dir, unit_normal_vector):

    # IDs for each rib label
    rib_ids = ["ALL", "T01R", "T02R", "T03R", "T04R", "T05R", "T06R", "T07R", "T08R", "T09R", "T10R", "T11R", "T12R",
                      "T01L", "T02L", "T03L", "T04L", "T05L", "T06L", "T07L", "T08L", "T09L", "T10L", "T11L", "T12L"] 
    
    labelled_coords = np.empty([0, 5])
    
    for i, rib_id in enumerate(rib_ids[1:]):
        fname = point_cloud_dir+"ribs_data_"+rib_id+".ipdata"
        with open(fname, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line_data = [float(j) for j in np.array(line.split()[:4])]
            line_data = np.hstack(([i+1], line_data))
            line_data[2:] = do_transform(unit_normal_vector, line_data[2:])
            labelled_coords = np.vstack((labelled_coords, line_data))

    return labelled_coords


###
def plot_labelled_coords(labelled_coords):

    unique_labels = np.unique(labelled_coords[:, 0])

    # plot in 3d
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colours = plt.cm.rainbow(np.linspace(0, 1, 24)) 
    for label in unique_labels: 
        col = colours[ int((label-1) - (label%2-1)*11 - (label>12)*11) ] # just splitting up colours for contrast
        #col[3] = 0.5
        label_indices = labelled_coords[:, 0]==label
        coords = labelled_coords[label_indices, 2:]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=col, zorder=3)
    plt.show()
    

###
def rotate_in_x(vector_1, ang):
    rotmat_x = [[ 1,           0,            0],
                [ 0, np.cos(ang), -np.sin(ang)],
                [ 0, np.sin(ang),  np.cos(ang)]]
    vector_2 = np.matmul(rotmat_x, vector_1)                
    return vector_2
    
    
###
def rotate_in_y(vector_1, ang):
    rotmat_y = [[  np.cos(ang), 0, np.sin(ang)],
                [            0, 1,           0],
                [ -np.sin(ang), 0, np.cos(ang)]]
    vector_2 = np.matmul(rotmat_y, vector_1)
    return vector_2
    

###
def rotate_in_z(vector_1, ang):
    rotmat_z = [[ np.cos(ang), -np.sin(ang), 0],
                [ np.sin(ang),  np.cos(ang), 0],
                [ 0,        0,               1]]
    vector_2 = np.matmul(rotmat_z, vector_1)
    return vector_2
    

###
def do_transform(unit_normal_vector, in_vector):

    # align to median plane
    angle_a = np.arctan(unit_normal_vector[2]/unit_normal_vector[0])
    vector_a = rotate_in_y(in_vector, angle_a)
    angle_b = -np.arcsin(unit_normal_vector[1]/1)
    vector_b = rotate_in_z(vector_a, angle_b)
    
    # correct z-direction
    unit_z = [0, 0, 1]
    unit_z_a = rotate_in_y(unit_z, angle_a)
    unit_z_b = rotate_in_z(unit_z_a, angle_b)
    correction_angle = np.arctan(unit_z_b[1]/unit_z_b[2])
    #unit_z_c = rotate_in_x(unit_z_b, correction_angle)
    vector_c = rotate_in_x(vector_b, correction_angle)
       
    return vector_c


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
torso_path = "/hpc/mpag253/Torso/"
input_list = np.array(pd.read_excel("/hpc/mpag253/Ribs/ribs_checklist.xlsx", skiprows=0, usecols=range(11)))

#############################################################################################################

print("\n")
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        run_ribs_prelim_fitting(input_list[i, 1:5], input_list[i, 5:], root, save_masks=False,
                              troubleshooting_images=[False, False])
        # check_masks(cohort, subject, condition, root)
print("\n")


