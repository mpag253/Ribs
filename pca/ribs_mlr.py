import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from sys import exit
import matplotlib.pyplot as plt
import joblib
import pickle

def print_out_ols_model(model):
    # print(model.summary())  # Print out the statistics
    print("="*78)
    print("\nModel overall p-value:      ", model.f_pvalue)
    print("Model overall r^2:            ", model.rsquared)
    print("Model overall r^2 (adjusted): ", model.rsquared_adj)
    print("\nModel parameters:\n", model.params.to_string())
    print("\nModel parameter standard errors:\n", model.bse.to_string())
    print("\nModel p-values:\n", model.pvalues.astype(float).to_string())
    print("\n", "="*78)
    return


########################################################################################################################
# SETUP
########################################################################################################################

n_mlr_modes = 4
score_headers = ["Subjects", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
depvars_list = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
all_sbj_data = pd.read_csv('input/pca_subject_data.csv', skiprows=1, delimiter=',')
sbj_scores = pd.read_csv('output/pca_weights_format.csv',
                         delimiter=',', header=None, usecols=range(n_mlr_modes+1),
                         names=score_headers[:n_mlr_modes+1])

print("\n"+"="*78+"\n\tSETUP\n"+"="*78+"\n")
print("All data:\n", all_sbj_data, "\n")     # does not have "P2BR"
print("Subject scores:\n", sbj_scores, "\n")   # has "P2BR"

# List of subjects for regression
sbj_list = sbj_scores.iloc[:, 0]
print("Subject list:\n", sbj_list, "\n")

# Find indices of subjects in subject data
sbj_idxs = np.empty(np.shape(sbj_list), dtype=int)
for s, sbj in enumerate(sbj_list):
    sbj = sbj.split('-')[-1]
    # sbj_idxs[s] = int(np.where(sbj == all_data[:, 0])[0][0])
    sbj_idxs[s] = int(np.where(sbj == all_sbj_data.iloc[:, 0])[0][0])

# Extract data for subjects
# sbj_data = all_data[sbj_idxs, :]
sbj_data = all_sbj_data.iloc[sbj_idxs, :].reset_index()
# print(sbj_data)

# Merge the two data frames
sbj_data = pd.concat((sbj_data, sbj_scores), axis=1)
print("Subject data:\n", sbj_data, "\n")

# Get the variable names
vnames = sbj_data.columns.values
print("Variable names:\n", vnames, "\n")


########################################################################################################################
# MLR (statsmodels)
########################################################################################################################
# The p-value for each term tests the null hypothesis that the coefficient is equal to zero (no effect).
# A low p-value (< 0.05) indicates that you can reject the null hypothesis.
print("\n"+"="*78+"\n\tGENERATING MLR MODEL\n"+"="*78+"\n")

mlr_method = 'auto' #'manual' #

if mlr_method == 'manual':
    
    # Need to specify the variable indices manually
    #model_var_indices = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    model_var_indices = [2, 3, 4, 5, 7, 8, 11, 13, 14] 
    
    depvars = depvars_list[:n_mlr_modes]
    mfs = [[] for _ in range(len(depvars))]
    models = [[] for _ in range(len(depvars))]
    for d, depvar in enumerate(depvars):
        mfs[d] = depvar+" ~ "+" + ".join(vnames[model_var_indices])
        models[d] = smf.ols(formula=mfs[d], data=sbj_data).fit()  
    
elif mlr_method == 'auto':

    # # Original method to find MLR model for each mode independently
    # depvar = "M5"
    # print("\n\n\n=== "+depvar+" ===\n\n")
    # model_var_indices = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    # complete = False
    # while not complete:
    #     mf = depvar+" ~ "+" + ".join(vnames[model_var_indices])
    #     print(mf)
    #     model = smf.ols(formula=mf, data=sbj_data).fit()
    #     # print(type(model1))  #statsmodels.regression.linear_model.RegressionResultsWrapper - OLS Regression Results
    #     # print(model1.summary())
    #     print_out_ols_model(model)
    #     maxp = model.pvalues.iloc[1:].idxmax()
    #     if model.pvalues[maxp] > 0.05:
    #         remidx = np.where(vnames == maxp)[0][0]
    #         model_var_indices.remove(remidx)
    #     else:
    #         complete = True
    
    # New method to find MLR model for each mode simultaneously
    depvars = depvars_list[:n_mlr_modes]
    model_var_indices = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    complete = False
    while not complete:
    
        print("Model independent variables:\n", vnames[model_var_indices])
        print("Model independent variable indices:\n", model_var_indices)
    
        # Pre-allocate array to store p-values
        pvals = np.empty([len(depvars), len(model_var_indices)])
    
        # Update MLR for each mode with current variables
        mfs = [[] for _ in range(len(depvars))]
        models = [[] for _ in range(len(depvars))]
        for d, depvar in enumerate(depvars):
            mfs[d] = depvar+" ~ "+" + ".join(vnames[model_var_indices])
            # print(mf)
            models[d] = smf.ols(formula=mfs[d], data=sbj_data).fit()
            # print(type(model1))  #statsmodels.regression.linear_model.RegressionResultsWrapper - OLS Regression Results
            # print(model1.summary())
            #print_out_ols_model(model)
            # print(model.pvalues.iloc[1:])
            pvals[d, :] = models[d].pvalues.iloc[1:]
    
        with np.printoptions(precision=3, suppress=True):
            print("Model p-values:\n", pvals)
    
        # Test p-values to either: (1) accept model and finish, or (2) remove variable and update model
        # test if all p-values > 0.05
        insig_pvals = (pvals > 0.05).astype(int)
        print("Insignificant p-values:\n", insig_pvals)
        insig_vars = np.all(insig_pvals, axis=0).astype(int)
        print("Universally insignificant independent variables:\n", insig_vars)
    
        num_insig_vars = np.sum(insig_vars)
        print("Number of universally insignificant variables:\t", num_insig_vars)
        # if any p-values are universally insignificant
        # remove the least significant of insignificant variables and update the model
        if num_insig_vars > 0:
            # find the least significant of the insignificant variables
            pval_mins = np.min(pvals, axis=0)
            with np.printoptions(precision=4, suppress=True):
                print("Minimum p-values for each predictor:\n", pval_mins)
            insig_vars_max_idx = np.argmax(pval_mins)
            print("insig_vars_max_idx:\t", insig_vars_max_idx)
            # remove the least significant of insignificant variables
            remove_var_num = model_var_indices[insig_vars_max_idx]
            model_var_indices.remove(remove_var_num)
            # ("complete" remains False)
            print("Removed least significant variable and updating model...\n")
        # else, accept the model
        else:
            complete = True
            print("\n"+"="*78+"\n\tACCEPTED MODEL\n"+"="*78+"\n")
            
# Display model info
print("Models:")
models_parameters = np.empty([len(depvars), len(model_var_indices)+1])
models_stderrs = np.empty([len(depvars), len(model_var_indices)+1])
models_pvalues = np.empty([len(depvars), len(model_var_indices)+1])
models_overall_pvalues = np.empty([len(depvars)])
models_overall_rsq = np.empty([len(depvars)])
models_overall_rsqadj = np.empty([len(depvars)])
for d, depvar in enumerate(depvars):
    print(mfs[d])
    models_parameters[d, :] = models[d].params
    models_stderrs[d, :] = models[d].bse
    models_pvalues[d, :] = models[d].pvalues.astype(float)
    models_overall_pvalues[d] = models[d].f_pvalue
    models_overall_rsq[d] = models[d].rsquared
    models_overall_rsqadj[d] = models[d].rsquared_adj
    #print_out_ols_model(models[d])
with np.printoptions(precision=3, suppress=True):
    print("\nOverall p-values:\n", models_overall_pvalues)
with np.printoptions(precision=2, suppress=True):
    print("\nOverall r^2:\n", models_overall_rsq)
    print("\nOverall r^2 (adjusted):\n", models_overall_rsqadj)
    print("\nParameters:\n", models_parameters)
    print("\nParameter standard errors:\n", models_stderrs)
with np.printoptions(precision=3, suppress=True):
    print("\nParameter p-values:\n", models_pvalues)
    
# Format as dict and save model
parameters_dict = {}
for d, depvar in enumerate(depvars):
    model_parameters_dict = {'Intercept': models_parameters[d, 0]}
    for v, var_index in enumerate(model_var_indices):
        model_parameters_dict[vnames[var_index]] = models_parameters[d, v+1]
    parameters_dict[depvar] = model_parameters_dict
mlr_dict = {'parameters': parameters_dict}
with open('output/mlr_model.pkl', 'wb') as f:
    pickle.dump(mlr_dict, f)
    
# Save as csv
np.savetxt('output/mlr_model_coeffs.csv', models_parameters, delimiter=',')
np.savetxt('output/mlr_model_pvalues.csv', models_pvalues, delimiter=',')
np.savetxt('output/mlr_model_rsquared.csv', models_overall_rsq, delimiter=',')


########################################################################################################################
# Manual approach to get conditional covariance matrix for modes
########################################################################################################################
# Gamma_(w|p) = Gamma_w - Gamma_(wp)*(Gamma_p)^(-1)*Gamma_(pw)
print("\n"+"="*78+"\n\tCONDITIONAL COVARIANCE\n"+"="*78+"\n")

# Set up data covariance matrix, Gamma_X
cov_indices = list(range(-n_mlr_modes, 0))+list(range(2, 12))+list(range(13, 18))
cov_data = sbj_data.iloc[:, cov_indices] # X transpose
cov_headers = cov_data.columns.values
mu_x = cov_data.mean(axis=0).transpose()
X = (cov_data - mu_x).transpose()
n_subjects = X.shape[1]
Gamma_X = 1/(n_subjects-0)*X.dot(X.transpose())  # EXPORT THIS!!!
# # Display
# print("\nCovariance headers:\n", cov_headers)
# print("\nX:\n", X)
# print("\nmu_x:\n", mu_x)
# print("\nGamma_X:\n", Gamma_X)
# print("\nN_subjects:\t", n_subjects)

# Define variables for conditional covariance
# weight indices (remove hard-code) --------
w_indices = list(range(n_mlr_modes))
# parameter indices
p_indices = [i for i, j in enumerate(cov_headers) if j in vnames[model_var_indices]]
# weight and parameter indices
wp_indices = w_indices+p_indices
print("Variables in input covariance:\n", cov_headers[wp_indices])

# Calculate conditional covariance
# mean of weights
mu_w = mu_x[w_indices].to_numpy()
mu_w = np.reshape(mu_w, [len(mu_w), 1])
# mean of parameters
mu_p = mu_x[p_indices].to_numpy()
mu_p = np.reshape(mu_p, [len(mu_p), 1])
# covariance sub-matrices
Gamma_w = Gamma_X.iloc[w_indices, w_indices].to_numpy()
Gamma_pw = Gamma_X.iloc[p_indices, w_indices]
Gamma_pw_labels = Gamma_pw.index.values
Gamma_pw = Gamma_pw.to_numpy()
Gamma_wp = Gamma_pw.transpose()
Gamma_p = Gamma_X.iloc[p_indices, p_indices].to_numpy()
# calculate gamma_(w|p) from sub-matrices
Gamma_wgivenp = Gamma_w - np.dot(np.dot(Gamma_wp, np.linalg.inv(Gamma_p)), Gamma_pw)

# Display
with np.printoptions(precision=4, suppress=True):
    print("\nGamma_w:\n", Gamma_w)
with np.printoptions(precision=2, suppress=True):
    print("\nGamma_p:\n", Gamma_X.iloc[p_indices, p_indices].to_numpy())
    print("\nGamma_pw:\n", Gamma_pw)
    print("\nGamma_wp:\n", Gamma_wp)
with np.printoptions(precision=4, suppress=True):
    print("\nGamma_w|p:\n", Gamma_wgivenp)
#np.savetxt("mlr_M{:d}_condcov_{}.csv".format(n_mlr_modes, pca_id), Gamma_wgivenp, delimiter=",")


########################################################################################################################
# Validate
########################################################################################################################
# (optional) check covariance matrix by reconstructing the mode equations
if True:
    print("\n"+"="*78+"\n\tVALIDATION\n"+"="*78+"\n")

    # (old)
    # (not sure what this variable was...)
    # mode_indices = range(len(wp_indices))
    # # this remains the same
    # for i in range(5):
    #     print("\nMode {:d}".format(i))
    #     premult = np.dot(Gamma_wp[i, mode_indices[i]],
    #                      np.linalg.inv(Gamma_p[np.ix_(mode_indices[i], mode_indices[i])]))  # (MxP)
    #     mode_coeffs = premult
    #     print("Coeffs:", mode_coeffs)
    #     mode_ints = mu_w[i] + np.dot(premult, -mu_p[mode_indices[i]])  # (5,1) + (5x14)*(14,1)
    #     print("Ints:  ", mode_ints)
    
    # (new)
    for i in range(n_mlr_modes):
        premult = np.dot(Gamma_wp[i, :], np.linalg.inv(Gamma_p))  # (MxP)
        mode_coeffs = premult
        mode_ints = mu_w[i] + np.dot(premult, -mu_p)  # (5,1) + (5x14)*(14,1)
        with np.printoptions(precision=4, suppress=True):
            print("Mode {:d}".format(i + 1))
            print("Int:  ", mode_ints)
            print("Coeffs:", mode_coeffs, "\n")
        
        
#########################################################################################################################
## Mesh conditional covariance matrix
#########################################################################################################################
#print("\n"+"="*78+"\n\tMESH CONDITIONAL COVARIANCE\n"+"="*78+"\n")

## load in manually generated mode shapes
#full_modeshapes = np.load('output/mode_shapes.npy')

## Generate full conditoinal covariance for weights 
#Gamma_wgivenp_full = np.eye(len(full_modeshapes.T))
#Gamma_wgivenp_full[:n_mlr_modes, :n_mlr_modes] = Gamma_wgivenp

## Calculte the mesh conditional covariance
##mesh_condcov = np.matmul(np.matmul(mlr_modeshapes, mlr_condcov), mlr_modeshapes.T)
#Gamma_ngivenp = np.matmul(np.matmul(full_modeshapes, Gamma_wgivenp_full), full_modeshapes.T)
## Save mesh conditional covariance
#np.save('output/pca_condcov.npy', Gamma_ngivenp)
#np.save('output/pca_condcov_wgivenp.npy', Gamma_wgivenp)

### Retrieve the mesh convariance (i.e. population sample covariance)
##pca = joblib.load('output/pca_model.sfeal')
##pca_cov = pca.get_covariance()

## Display
#with np.printoptions(precision=1, suppress=True):
    #print('Gamma_n|p:', np.shape(Gamma_ngivenp))
    #print(Gamma_ngivenp, '\n')
    #print('Gamma_n:', np.shape(pca_cov))
    #print(pca_cov, '\n')


########################################################################################################################
# Correlation plots
########################################################################################################################
if True:
    print("\n"+"="*78+"\n\tCORRELATION PLOTS\n"+"="*78+"\n")

    for d, depvar in enumerate(depvars):
        params = models[d].params.index.values
        males = sbj_data.loc[:, 'Sex'] == 0
        mean_vals = sbj_data.loc[:, params[1:]].mean()
        mean_vals_m = sbj_data.loc[males, params[1:]].mean()
        mean_vals_f = sbj_data.loc[~males, params[1:]].mean()
        n_params = len(params)-1  # ignoring intercept
        subplot_dim_h = 3 #int((n_params+1)/2)
        subplot_dim_w = 4
        fig, axs = plt.subplots(subplot_dim_h, subplot_dim_w)
        for p in range(1, n_params+1):
            ax = axs[int((p-1)/subplot_dim_w), (p-1)%subplot_dim_w]
            # Plot male and female data
            ax.scatter(sbj_data.loc[males,  params[p]], sbj_data.loc[males,  depvar], color="blue")
            ax.scatter(sbj_data.loc[~males, params[p]], sbj_data.loc[~males, depvar], color="red")
            # Plot overall regression line
            rline_x = np.linspace(sbj_data.loc[:,params[p]].min(), sbj_data.loc[:,params[p]].max(), 2)
            rline_y = models[d].params.loc['Intercept'] + models[d].params.loc[params[p]]*rline_x
            for param in mean_vals.index.values:
                if param != params[p]:
                    rline_y += models[d].params.loc[param]*mean_vals.loc[param]
            ax.plot(rline_x, rline_y, color="black")
            # Plot sex-based regression lines
            rline_xm = np.linspace(sbj_data.loc[males, params[p]].min(), sbj_data.loc[males, params[p]].max(), 2)
            rline_xf = np.linspace(sbj_data.loc[~males, params[p]].min(), sbj_data.loc[~males, params[p]].max(), 2)
            rline_ym = models[d].params.loc['Intercept'] + models[d].params.loc[params[p]]*rline_xm
            rline_yf = models[d].params.loc['Intercept'] + models[d].params.loc[params[p]]*rline_xf
            for param in mean_vals.index.values:
                if param != params[p]:
                    rline_ym += models[d].params.loc[param]*mean_vals_m.loc[param]
                    rline_yf += models[d].params.loc[param]*mean_vals_f.loc[param]
            ax.plot(rline_xm, rline_ym, color="black", linestyle="dotted")
            ax.plot(rline_xf, rline_yf, color="black", linestyle="dashed")
            # Add labels to plot
            if models_pvalues[d, p] < 0.05:
                ax.set_xlabel("{}* (p={:.3f})".format(params[p], models_pvalues[d, p]), fontweight='bold')
            else:
                ax.set_xlabel("{} (p={:.3f})".format(params[p], models_pvalues[d, p]))
                ax.set_facecolor([.95, .95, .95])
            ax.set_ylabel(depvar)
        # if n_params < subplot_dim_w*subplot_dim_h:
        #     fig.delaxes(axs[subplot_dim_h-1, subplot_dim_w-1])
        plt.tight_layout()
    print('Plotting correlations...', end="\r")
    plt.show()
    print('Plotting correlations... Finished.\n')




