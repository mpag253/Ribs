import numpy as np
import matplotlib.pyplot as plt
#import sparse
#import pickle as pkl
#import pandas as pd
#from scipy import interpolate
import scipy.optimize as spo
#import os
import copy
from sys import exit
from sklearn import decomposition

def get_cartesian(nData_polar):
    nData_cart = copy.deepcopy(nData_polar)
    x = np.multiply(nData_polar[:,1], np.sin(nData_polar[:,0]))
    y = np.multiply(nData_polar[:,1], np.cos(nData_polar[:,0]))
    nData_cart[:,0] = x
    nData_cart[:,1] = y
    return nData_cart

fit_tss = np.load('../fitting/output/ribs_fit_data_t.npy')
fit_rss = np.load('../fitting/output/ribs_fit_data_r.npy')
fit_zss = np.load('../fitting/output/ribs_fit_data_z.npy')

n_sbj = np.shape(fit_tss)[0]
n_dof = np.shape(fit_tss)[1]

nData_ref_polar = np.vstack((fit_tss[0,:],fit_rss[0,:],fit_zss[0,:])).T
nData_ref = get_cartesian(nData_ref_polar)

print(n_sbj)
print(np.shape(nData_ref))


def do_rotate_ndata_x(nData_, angle_x):
    rotmat = np.array([[np.cos(angle_x), -np.sin(angle_x)],
                       [np.sin(angle_x),  np.cos(angle_x)]])
    nData_[:, [1, 2]] = np.matmul(nData_[:, [1, 2]], rotmat)
    return nData_

def do_shift_ndata_x(nData_, shift_x):
    nData_[:, 0] += shift_x
    return nData_

def do_shift_ndata_y(nData_, shift_y):
    nData_[:, 1] += shift_y
    return nData_

def do_shift_ndata_z(nData_, shift_z):
    nData_[:, 2] += shift_z
    return nData_

def do_scale_ndata(nData_, scale):
    nData_ *= scale
    return nData_

def do_transforms(nData_sbj,x):
    nData_sbj_new = copy.deepcopy(nData_sbj)
    nData_sbj_new = do_rotate_ndata_x(nData_sbj_new, x[0])
    nData_sbj_new = do_shift_ndata_y(nData_sbj_new, x[1])
    nData_sbj_new = do_shift_ndata_z(nData_sbj_new, x[2])
    nData_sbj_new = do_scale_ndata(nData_sbj_new, x[3])
    return nData_sbj_new

def get_rmse(nData_sbj_tf,nData_ref):
    rmse = np.sqrt(np.mean(np.square(nData_sbj_tf-nData_ref)))
    #print(rmse)
    return rmse

def min_func(x,nData_sbj,nData_ref):
    nData_sbj_tf = do_transforms(nData_sbj,x)
    metric = get_rmse(nData_sbj_tf,nData_ref)
    return metric


all_data_tf = np.empty([n_sbj,n_dof*3])
all_data_tf[0,:] = np.hstack((nData_ref[:,0],nData_ref[:,1],nData_ref[:,2])).T
for sbj in range(1,n_sbj):
    
    nData_sbj_polar = np.vstack((fit_tss[sbj,:],fit_rss[sbj,:],fit_zss[sbj,:])).T
    nData_sbj = get_cartesian(nData_sbj_polar)   
    
    res = spo.minimize(min_func,
                       [0.0,0.0,0.0,1.0],
                       args=(nData_sbj,nData_ref),
                       method='Nelder-Mead',
                       options={'maxiter':100000})
    # print(res)
    print("\nSuccess?", res.success)
    print("RMSE:", res.fun)
    print("X:", res.x)
    resx = res.x

    nData_sbj_tf = do_transforms(nData_sbj, resx)
    all_data_tf[sbj,:] = np.hstack((nData_sbj_tf[:,0],nData_sbj_tf[:,1],nData_sbj_tf[:,2])).T
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')    
    #ax.scatter(nData_sbj_tf[:,0], nData_sbj_tf[:,1], nData_sbj_tf[:,2])
    #ax.scatter(nData_ref[:,0], nData_ref[:,1], nData_ref[:,2])
    #plt.show()    
    
    
# Get mean
data_mean = np.mean(all_data_tf, axis=0)
print(data_mean)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')    
#ax.scatter(data_mean[0:n_dof], data_mean[n_dof:2*n_dof], data_mean[2*n_dof:3*n_dof])
#plt.show()  

# Centre data
data_centred = all_data_tf - data_mean


# PCA
X = data_centred
n_modes = n_sbj-1
pca = decomposition.PCA(n_components=n_modes)
pca.fit(X)
pca_mean = pca.mean_
components = pca.components_.T
singular_values = pca.singular_values_
#explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

#print(pca.mean_)
#print(pca.components_.T)
print(100*explained_variance_ratio[0:5])
print(np.cumsum(100*explained_variance_ratio[0:5]))


#
# SCORES/WEIGHTS
#
mode_scores = np.empty([n_sbj,n_modes])
for sbj in range(n_sbj):
    subject = X[sbj,:] - pca_mean
    #print(np.shape(subject))
    #print(np.shape(components))
    scores = np.dot(subject, components)
    #print(scores)
    #print(np.shape(scores))
    #mode_scores.append(scores)
    mode_scores[sbj,:] = scores
    
scores_sd = np.std(mode_scores, axis=0)
scores_mean = np.mean(mode_scores, axis=0)
#print(scores_mean)

mode_weights = np.empty([n_sbj,n_modes])
#print(np.shape(scores_sd))
#print(np.shape(mode_scores))
for sbj in range(n_sbj):
    mode_weights[sbj,:] = np.divide(mode_scores[sbj,:], scores_sd)


#print(mode_weights)
#np.savetxt("output/pca_weights.csv", mode_weights, delimiter=",")



#
# PLOT MODES
#

def plot_rib_lines(nData,ax,col):
    for i in range(10):
        #print(np.shape(nData))
        #print(0*n_dof+10*i)
        #print(1*n_dof+10*(i+1))
        #print(1*n_dof+10*i)
        #print(2*n_dof+10*(i+1))
        #print(2*n_dof+10*i)
        #print(3*n_dof+10*(i+1))
        ax.plot(nData[(0*n_dof+10*i):(0*n_dof+10*(i+1))], 
                nData[(1*n_dof+10*i):(1*n_dof+10*(i+1))], 
                nData[(2*n_dof+10*i):(2*n_dof+10*(i+1))],
                c=col)

#print(np.shape(scores_sd))

for mode in range(4):
    
    #print(np.shape(components))
    #print(np.shape(singular_values))
    #print(singular_values[mode])
          
    mode_data_pos = data_mean + (2.5*scores_sd[mode]*components[:,mode])
    mode_data_neg = data_mean - (2.5*scores_sd[mode]*components[:,mode])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_rib_lines(data_mean,ax,'blue')
    plot_rib_lines(mode_data_pos,ax,'orange')
    plot_rib_lines(mode_data_neg,ax,'green')
    #ax.scatter(data_mean[0:n_dof], data_mean[n_dof:2*n_dof], data_mean[2*n_dof:3*n_dof])
    #ax.scatter(mode_data_pos[0:n_dof], mode_data_pos[n_dof:2*n_dof], mode_data_pos[2*n_dof:3*n_dof])
    #ax.scatter(mode_data_neg[0:n_dof], mode_data_neg[n_dof:2*n_dof], mode_data_neg[2*n_dof:3*n_dof])
    
plt.show()      







