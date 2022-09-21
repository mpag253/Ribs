import numpy as np
import matplotlib.pyplot as plt
import sparse
import pickle as pkl
import pandas as pd
from scipy import interpolate
import os
import copy
from sys import exit


def run_icspace_fitting(input_info, image_info, root, torso_path):
    """ Main script to ...
    """
    
    # unpack
    [cohort, subject, condition] = input_info
    #[d1, d2, d3, p1, p2, p3] = image_info
    print("---Ribs fitting... "+subject+" "+condition, end="\r")
    
    # Load the rib point clouds - transformed in aligned reference frame
    point_cloud_dir = '/hpc/mpag253/Ribs/point_clouds/output/'+cohort+"/"+subject+"/"+condition+"/Ribs/Data/"
    median_dir = '/hpc/mpag253/Ribs/median_plane/output/'+cohort+"/"+subject+"/"+condition+'/Median_Plane/'
    # load the median plane
    median_plane = np.load(median_dir + "median_plane.npy")
    # calculate the reorientation vector
    normal_vector = median_plane[:3]
    unit_normal_vector = normal_vector/np.linalg.norm(median_plane[:3]) 
    
    # load the point clouds (RIBS 4, 5, and 6 ONLY) and transform
    rib_ids = ["T02R", "T03R", "T04R", "T05R", "T06R",
               "T02L", "T03L", "T04L", "T05L", "T06L"]#, "T07R", "T08R", "T09R", "T10R"#, "T07L", "T08L", "T09L", "T10L"] # IDs for each rib label  
    labelled_coords = load_point_clouds(rib_ids, point_cloud_dir, unit_normal_vector)
    
    # Load the landmarks
    fname = torso_path+"landmarks/"+cohort+"/"+subject+"/"+condition+"/landmarks.txt"
    with open(fname, "r") as f:
        lines = f.readlines()
    landmarks = [float(i) for i in lines]

    # Trim
    trim_left = landmarks[2] + 35
    trim_rght = landmarks[2] - 35
    keep_indices = (labelled_coords[:, 2] > trim_left) + (labelled_coords[:, 2] < trim_rght)
    coords_trimmed = labelled_coords[keep_indices]
    #plot_labelled_coords(coords_trimmed)
    
    ## Define centre
    #centre_y = (np.max(coords_trimmed[:, 3])+np.min(coords_trimmed[:, 3]))/2
    #centre_xy = [landmarks[2], centre_y]
    
    ## Convert to cylindrical coordinates
    #coords_cyl = copy.deepcopy(coords_trimmed)
    ##coords_cylindrical[:, 2:4] -= centre_xy
    ##plot_labelled_coords(coords_cylindrical)
    #xy_prime = coords_trimmed[:, 2:4] - centre_xy
    #angles = np.arctan2(xy_prime[:, 0], xy_prime[:, 1])
    #radii = np.sqrt(np.square(xy_prime[:, 0]) + np.square(xy_prime[:, 1]))
    #coords_cyl[:, 2] = angles
    #coords_cyl[:, 3] = radii
           
    # Find the z-value of the intercostal space
    # intercostal space numbers correspond with the superior rib number
    # (i.e. ic space 4 is between 4th and 5h ribs)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    rib_zs = np.empty([6])
    #print(np.unique(coords_trimmed[:, 0]))
    #theta_mins = [0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #theta_maxs = [2.4, 2.4,  2.4, 2.4, 2.4, 2.2, 2.0, 1.8, 1.5, 2.4, 2.4,  2.4, 2.4, 2.4, 2.2, 2.0, 1.8, 1.5]
    theta_mins = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    theta_maxs = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4]   
    fit_ts = np.empty([0])
    fit_rs = np.empty([0])
    fit_zs = np.empty([0])
    for i, rib_id in enumerate(np.unique(coords_trimmed[:, 0])):
        
        rib_indices = coords_trimmed[:, 0]==rib_id
        coords_cart = coords_trimmed[rib_indices, 2:]
        
        # Define centre
        centre_y = (np.max(coords_cart[:, 1])+np.min(coords_cart[:, 1]))/2
        centre_xy = [landmarks[2], centre_y]
        
        # Convert to cylindrical coordinates
        coords_cyl = copy.deepcopy(coords_cart)
        #plot_labelled_coords(coords_cylindrical)
        xy_prime = coords_cart[:, 0:2] - centre_xy
        angles = np.arctan2(xy_prime[:, 0], xy_prime[:, 1])
        radii = np.sqrt(np.square(xy_prime[:, 0]) + np.square(xy_prime[:, 1]))
        coords_cyl[:, 0] = angles
        coords_cyl[:, 1] = radii        
        
        # get rib coords
        rib_coords = coords_cyl
        # sort rib coords
        sort_indices = np.argsort(rib_coords[:, 0])
        rib_coords = rib_coords[sort_indices, :]
        
        # Polynomial fit to z
        polyfit_z = np.polyfit(rib_coords[:, 0], rib_coords[:, 2], deg=3)
        poly1d_z = np.poly1d(polyfit_z)
        
        # Polynomial fit to r
        polyfit_r = np.polyfit(rib_coords[:, 0], rib_coords[:, 1], deg=3)
        poly1d_r = np.poly1d(polyfit_r)
        
        # Get polynomial fit 
        #theta_min = 0.4
        #theta_max = 2.4
        theta_min = np.min(np.abs(rib_coords[:, 0]))
        theta_max = np.max(np.abs(rib_coords[:, 0]))
        if rib_id in range(1,6):
            thetas = np.linspace(theta_min, theta_max, 10)
            #thetas = np.arange(theta_mins[i], theta_maxs[i], 0.1)
            #thetas = np.linspace(theta_mins[i], theta_maxs[i], 10)
        elif rib_id in range(6,11):
            thetas = -np.linspace(theta_min, theta_max, 10) 
            #thetas = -np.arange(theta_mins[i], theta_maxs[i], 0.1)
            #thetas = -np.linspace(theta_mins[i], theta_maxs[i], 10)
        
        # Convert to cartesian
        fit_r = poly1d_r(thetas)
        fit_z = poly1d_z(thetas)
        fit_x = np.multiply(fit_r, np.sin(thetas)) + centre_xy[0]
        fit_y = np.multiply(fit_r, np.cos(thetas)) + centre_xy[1]
        
        fit_ts = np.hstack((fit_ts, thetas))
        fit_rs = np.hstack((fit_rs, fit_r))
        fit_zs = np.hstack((fit_zs, fit_z))
        
        ## Plot
        #ax.plot(fit_x, fit_y, fit_z)
        #ax.scatter(centre_xy[0], centre_xy[1], np.mean(rib_coords[:, 2]), c='black')
    
    #ax.scatter(coords_trimmed[:, 2], coords_trimmed[:, 3], coords_trimmed[:, 4])
    #plt.show()
    
    ## Load the fitted torso mesh and define IC45 relative to mesh
    #mesh_path = torso_path+"surface_fitting/output/"+subject+"/"+condition+"/Torso/"
    #file_data = open(mesh_path+"Torso_fitted.exnode", 'r')
    #lines = file_data.readlines()
    #file_data.close()
    #for ln, line in enumerate(lines):
        #if line.startswith(" Node: "):
            #node_num = int(line.split()[-1])
            #if node_num == 97:
                #torso_zmin = float(lines[ln+3].split()[0])
            #elif node_num == 208:
                #torso_zmax = float(lines[ln+3].split()[0])
    #rib_mesh_positions = (rib_zs - torso_zmin)/(torso_zmax - torso_zmin)
    
    ## Save to ...
    #fdir = '/hpc/mpag253/Ribs/fitting/output/'+cohort+"/"+subject+"/"+condition+"/"
    #fname = "mesh_ribs_positions_t4to6.txt"
    #np.savetxt(fdir+fname, rib_mesh_positions)
    
    print("---Ribs fitting... "+subject+" "+condition+"\t Done.")
    #with np.printoptions(precision=2, suppress=True):
        #print("   ", rib_mesh_positions, "\n")
    
    #return rib_mesh_positions
    
    return fit_ts, fit_rs, fit_zs
    

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
def load_point_clouds(rib_ids, point_cloud_dir, unit_normal_vector):
               
    labelled_coords = np.empty([0, 5])
    
    for i, rib_id in enumerate(rib_ids):
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
fit_tss = np.empty([0, 100])
fit_rss = np.empty([0, 100])
fit_zss = np.empty([0, 100])
for i in range(np.shape(input_list)[0]):
    if input_list[i, 0] == 1:
        [fit_ts, fit_rs, fit_zs] = run_icspace_fitting(input_list[i, 1:4], input_list[i, 5:], root, torso_path)
        fit_tss = np.vstack((fit_tss, fit_ts))
        fit_rss = np.vstack((fit_rss, fit_rs))
        fit_zss = np.vstack((fit_zss, fit_zs))
        #print(np.shape(fit_rss))
        #print(np.shape(fit_zss))
np.save('output/ribs_fit_data_t.npy', fit_tss)
np.save('output/ribs_fit_data_r.npy', fit_rss)
np.save('output/ribs_fit_data_z.npy', fit_zss)

