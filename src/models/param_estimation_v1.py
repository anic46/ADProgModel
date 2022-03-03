import pandas as pd
import numpy as np


class ColumnNames:

    def __init__(self, subid, time, age, cogn, r1_mri, r2_mri, r1_av45, r2_av45):
        self.subid = subid
        self.time = time
        self.age = age
        self.cogn = cogn
        self.r1_mri = r1_mri
        self.r2_mri = r2_mri
        self.r1_av45 = r1_av45
        self.r2_av45 = r2_av45
        


class DTIMat:
    
    def __init__(self, adjmat):
        self.adjmat = adjmat
        self.laplacian = np.diag(np.squeeze(np.asarray(self.adjmat.sum(axis=1)))) - self.adjmat
        self.eigval, self.eigvec = np.linalg.eig(self.laplacian)


def estimate_beta_tpo_alternating_min(datdf, dfcolnames, dtiinfo):
    '''Compute best beta and tpo alternately by fixing the other.
    Inputs- datdf: Dataframe includes columns with AV45 and years info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dttinfo: object of class DTIMat which has the DTI adjacency matrix, its laplacian, and eigens
    
    Output - beta_estm_new: estimate of beta
    tpo_estm_new: estimate of time post-onset 
    '''
    
    beta_estm_old = 0.05
    tpo_estm_old = np.Inf
    max_iters = 20

    beta_tpo_estm_list = []

    while_flag = 1
    itercount = 0

    while while_flag == 1:

        # compute best tpo for the given value of beta
        tpo_estm_new, tpolist, objfunclist = find_best_tpo_corr(datdf, dfcolnames, dtiinfo, beta_estm_old)

        # recompute beta for the best tpo value
        beta_estm_new = estimate_beta_given_tpo(datdf, dfcolnames, dtiinfo, beta_estm_old, tpo_estm_new)

        # append estimates to the list
        beta_tpo_estm_list.append([(beta_estm_new, tpo_estm_new)])
        itercount += 1

        # update the parameters of the while loop
        if itercount > max_iters:
            while_flag = 0
        else:
            beta_estm_old = beta_estm_new
            tpo_estm_old = tpo_estm_new
            
    return beta_estm_new, tpo_estm_new



def compute_H_tilda_mat(dtiinfo, beta_val, tpo_val, time_since_tpo):
    '''Compute H tilda matrix from Raj 2015 paper. It depends on the time since onset of
    the disease and the eigvenvalues and vectors of the laplacian matrix.
    Inputs - dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_val: value of beta parameter
    tpo_val: value of time post-onset parameter (baseline measurement)
    time_since_tpo: time since baseline measurement
    
    Outputs - H_tilda_betat_mat: H tilda matrix
    '''
    diagonal_mat = np.zeros(dtiinfo.eigval.shape)

    for ii in range(len(dtiinfo.eigval)):

        eigval = dtiinfo.eigval[ii]
        t_org = tpo_val + time_since_tpo # time since origin

        if eigval == 0:
            diagonal_mat[ii] = 1/(beta_val*t_org)
        else:
            expont = np.exp(-eigval*beta_val*t_org)
            diagonal_mat[ii] = eigval*expont/(1-expont)

    diagonal_mat = np.diag(diagonal_mat)

    H_tilda_betat_mat = dtiinfo.eigvec @ diagonal_mat @ dtiinfo.eigvec.T
    
    return H_tilda_betat_mat



def compute_H_tilda_mat_warm_start(dtiinfo, beta_val, tpo_val, time_since_tpo):
    '''Compute H tilda matrix from Raj 2015 paper. It depends on the time since onset of
    the disease and the eigvenvalues and vectors of the laplacian matrix.
    NOTE: In this, the amount of total amyloid (phi) at time zero is x_0.
    
    Inputs - dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_val: value of beta parameter
    tpo_val: value of time post-onset parameter (baseline measurement)
    time_since_tpo: time since baseline measurement
    
    Outputs - H_tilda_betat_mat: H tilda matrix
    '''
    diagonal_mat = np.zeros(dtiinfo.eigval.shape)

    for ii in range(len(dtiinfo.eigval)):

        eigval = dtiinfo.eigval[ii]
        t_org = tpo_val + time_since_tpo # time since origin
            
        if eigval == 0:
            diagonal_mat[ii] = 1/(beta_val*(t_org+1))
        else:
            expont = np.exp(-eigval*beta_val*t_org)
            diagonal_mat[ii] = eigval*expont/(1-expont +eigval*beta_val)

    diagonal_mat = np.diag(diagonal_mat)

    H_tilda_betat_mat = dtiinfo.eigvec @ diagonal_mat @ dtiinfo.eigvec.T
    
    return H_tilda_betat_mat



def find_best_tpo_corr(datdf, dfcolnames, dtiinfo, beta_curval):
    ''' Find the best time post onset value for given estimate of beta.
    Use correlation between dphi/dt and H_tilda x phi for the computation. 
    See Raj 2015 Cell Reports for details. Performs grid search.
    
    Inputs - 
    datdf: dataframe with av45, years, and baseline age of subject info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_curval: value of beta parameter
    
    Outputs - 
    tpo_best_val: best estimate for tpo
    tpo_val_list: values of tpo used in grid search
    corval_list: value of objective function (correlations) for tpo values
    '''
    
    max_tpoval = np.floor(datdf[dfcolnames.age].min() - 50) # deposition starts around 50 years of age
    tpo_val_list = np.arange(0.2, max_tpoval, step=0.2)
    corval_list = np.zeros(tpo_val_list.shape)

    for ii in range(len(tpo_val_list)):

        corval_list[ii] = correlation_phi_fut_cur(datdf, dfcolnames, dtiinfo,
                                                  beta_cur=beta_curval, tpo_cur=tpo_val_list[ii])
    
    # find time where objective function achieve maximum value
    tpo_best_est = tpo_val_list[np.argmax(corval_list)]
    
    return tpo_best_est, tpo_val_list, corval_list



def find_best_tpo_mse(datdf, dfcolnames, dtiinfo, beta_curval):
    ''' Find the best time post onset value for given estimate of beta.
    Use mean squared error between dphi/dt and beta x H_tilda x phi for the computation. 
    See Raj 2015 Cell Reports for details. Performs grid search.
    
    Inputs - 
    datdf: dataframe with av45, years, and baseline age of subject info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_curval: value of beta parameter
    
    Outputs - 
    tpo_best_est: best estimate for tpo
    tpo_val_list: values of tpo used in grid search
    errorval_list: value of objective function (correlations) for tpo values
    '''
    
    max_tpoval = np.floor(datdf[dfcolnames.age].min() - 50) # deposition starts around 50 years of age
    tpo_val_list = np.arange(0.5, max_tpoval, step=0.5)
    errorval_list = np.zeros(tpo_val_list.shape)

    for ii in range(len(tpo_val_list)):

        errorval_list[ii] = mse_phi_fut_cur(datdf, dfcolnames, dtiinfo, 
                                            beta_cur=beta_curval, tpo_cur=tpo_val_list[ii])
    
    # find time where objective function achieve minimum value
    tpo_best_est = tpo_val_list[np.argmin(errorval_list)]

    return tpo_best_est, tpo_val_list, errorval_list



def mse_phi_fut_cur(datdf, dfcolnames, dtiinfo, beta_cur, tpo_cur):
    
    ''' Compute the squared error between dphi/dt and beta x H_tilda x phi.
    Use all pairs of consecutive time points to compute dphi/dt and
    sum over them to get total error
    
    Inputs - 
    datdf: dataframe with av45, years, and baseline age of subject info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_cur: value of beta parameter used in computing H tilda
    tpo_cur: value of time post onset parameter used in computing H tilda
    
    Outputs - 
    error_val: error between observed and predicted values of dphi/dt
    '''
    
    error_val = 0

    # pairs of time used for the computation
    time_pairs_list = zip(datdf[dfcolnames.time].values[:-1], datdf[dfcolnames.time].values[1:])

    # repeat computation for all time points and sum the value
    for time1, time2 in time_pairs_list:

        # get the phi and deta phi values
        phi_time_cur = datdf.loc[datdf[dfcolnames.time] == time1, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values[0]
        phi_time_next = datdf.loc[datdf[dfcolnames.time] == time2, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values[0]
        delta_time = time2 - time1

        delta_phi_by_delta_t = (phi_time_next - phi_time_cur)/delta_time

        # compute H tilda matrix and the corresponding product
        H_tilda_mat = compute_H_tilda_mat_warm_start(dtiinfo, beta_val=beta_cur, 
                                                     tpo_val=tpo_cur, time_since_tpo=time1)

        H_tilda_times_phi_t = H_tilda_mat @ phi_time_cur

        # compute mse difference between them them to be unit vectors
        error_val += np.linalg.norm(delta_phi_by_delta_t - beta_cur*H_tilda_times_phi_t)**2
    
    # return the summed value
    return error_val    



def correlation_phi_fut_cur(datdf, dfcolnames, dtiinfo, beta_cur, tpo_cur):
    
    ''' Compute the correlation between dphi/dt and H_tilda x phi.
    Use all pairs of consecutive time points to compute dphi/dt and
    sum over them to get total correlation.
    
    Inputs - 
    datdf: dataframe with av45, years, and baseline age of subject info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_cur: value of beta parameter used in computing H tilda
    tpo_cur: value of time post onset parameter used in computing H tilda
    
    Outputs - 
    correlation_val: correlation between observed and predicted direction of dphi/dt
    '''
    
    correlation_val = 0

    # pairs of time used for the computation
    time_pairs_list = zip(datdf[dfcolnames.time].values[:-1], datdf[dfcolnames.time].values[1:])

    # repeat computation for all time points and sum the value
    for time1, time2 in time_pairs_list:

        # get the phi and deta phi values
        phi_time_cur = datdf.loc[datdf[dfcolnames.time] == time1, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values[0]
        phi_time_next = datdf.loc[datdf[dfcolnames.time] == time2, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values[0]
        delta_time = time2 - time1

        delta_phi_by_delta_t = (phi_time_next - phi_time_cur)/delta_time

        # compute H tilda matrix and the corresponding product
        H_tilda_mat = compute_H_tilda_mat(dtiinfo, beta_val=beta_cur, 
                                          tpo_val=tpo_cur, time_since_tpo=time1)

        H_tilda_times_phi_t = H_tilda_mat @ phi_time_cur

        # normalize them to be unit vectors
        delta_phi_by_delta_t = delta_phi_by_delta_t / np.linalg.norm(delta_phi_by_delta_t)
        H_tilda_times_phi_t = H_tilda_times_phi_t / np.linalg.norm(H_tilda_times_phi_t)

        # take their dot product
        correlation_val += np.dot(delta_phi_by_delta_t, H_tilda_times_phi_t.T)[0,0]

    # return the summed value
    return correlation_val    



def estimate_beta_given_tpo(datdf, dfcolnames, dtiinfo, beta_oldestm, tpo_cur):
    
    '''Compute the best beta given the current value of tpo and the old value of beta.
    Use all pairs of consecutive time points.
    
    Inputs - 
    datdf: dataframe with av45, years, and baseline age of subject info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    beta_oldestm: old value of beta parameter
    tpo_cur: value of time post onset parameter
    
    Outputs - 
    beta_newestm: updated estimate of beta
    '''
    
    # pairs of time used for the computation
    time_pairs_list = zip(datdf[dfcolnames.time].values[:-1], datdf[dfcolnames.time].values[1:])

    numerator = 0
    denominator = 0

    # repeat computation for all time points and sum the value
    for time1, time2 in time_pairs_list:

        # get the phi and deta phi values
        phi_time_cur = datdf.loc[datdf[dfcolnames.time] == time1, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values[0]
        phi_time_next = datdf.loc[datdf[dfcolnames.time] == time2, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values[0]
        delta_time = time2 - time1

        delta_phi_by_delta_t = (phi_time_next - phi_time_cur)/delta_time

        # compute H tilda matrix and the corresponding product
        H_tilda_mat = compute_H_tilda_mat(dtiinfo, beta_val=beta_oldestm, tpo_val=tpo_cur, time_since_tpo=time1)

        y_tilda_t = H_tilda_mat @ phi_time_cur

        # numerator and denominator dot products
        numerator += np.dot(y_tilda_t, delta_phi_by_delta_t)[0,0]
        denominator += np.linalg.norm(y_tilda_t)**2

        beta_newestm = numerator/denominator
    
    return beta_newestm



def estimate_beta_from_AV45(mydatdf, dfcolnames, dtiinfo):
    '''Estimate beta value directly from AV45 without estimating tpo. Computation
    is similar to squared error minimization of linear function. Since phi values
    are in the data, the difference between consecutive time points is first calculated
    to compute D (same as x in Raj 2015 Cell Reports). This function handles
    multiple subjects.
    NOTE: mydatdf should only contain rows where AV45 data is available
    
    Inputs - 
    mydatadf: dataframe with av45, years, and baseline age of subject info
    dfcolnames: object of class ColumnNames with datdf colnames as attributes
    dtiinfo: object of class DTIMat which includes adjacency matrix of DTI, laplacian and eigs
    
    Outputs - 
    beta_estm: estimate of beta
    '''
    
    numerator = 0
    denominator = 0

    # sum over all subjects in the dataframe
    for ridval in mydatdf[dfcolnames.subid].unique():    
        
        # extract the information for a given subject
        datdf = mydatdf.loc[mydatdf[dfcolnames.subid]==ridval, 
        [dfcolnames.time, dfcolnames.r1_av45, dfcolnames.r2_av45]].diff().copy()
        datdf.dropna(inplace=True) # remove the first row after diff which is filled with NaNs

        
        for idx1, idx2 in zip(datdf.index[:-1], datdf.index[1:]):
            D_time_cur = datdf.loc[idx1,[dfcolnames.r1_av45, dfcolnames.r2_av45]].values
            D_time_next = datdf.loc[idx2, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values
            delta_T = datdf.loc[idx2, dfcolnames.time] # note that since we took diff already, this is the time since last update

            # change in D
            delta_D_by_delta_T = (D_time_next - D_time_cur)/delta_T
            H_times_D_t = np.asarray(dtiinfo.laplacian @ D_time_cur)[0]

            # inversion computations
            numerator += np.dot(H_times_D_t, delta_D_by_delta_T)
            denominator += np.dot(H_times_D_t, H_times_D_t)

    beta_estm = -numerator/denominator
    
    return beta_estm


def estimate_alpha_gamma_wo_Y_df(fuldatdf, dfcolnames):
    '''Estimate alpha1, alpha2, and gamma when functional measurement
    is unavailable in the data. alpha1 and alpha2*gamma is computed.
    See derivation for the formula. This function handles multiple
    subjects.
    
    Inputs -
    datdf: dataframe consisting of mri, av45, cognition scores, and years measurements
    dfcolnames: object of class ColumnNames with attributes as column names of datdf
    
    Output - 
    alpha2_gamma: estimated product of alpha2 and gamma
    alpha1: estimated alpha1
    '''
    
    K1 = 0
    K2 = 0
    K3 = 0
    K4 = 0
    K5 = 0

    for ridval in fuldatdf[dfcolnames.subid].unique():

        datdf = fuldatdf.loc[fuldatdf[dfcolnames.subid]==ridval].copy()

        for idx1, idx2, in zip(datdf.index[:-1], datdf.index[1:]):

            # collecting the variables required for computation
            C_t = datdf.loc[idx1, dfcolnames.cogn]

            X_vec_t = datdf.loc[idx1,[dfcolnames.r1_mri, dfcolnames.r2_mri]].values

            delta_X_vec_t = datdf.loc[idx2, [dfcolnames.r1_mri, dfcolnames.r2_mri]].values - X_vec_t
            delta_T = datdf.loc[idx2, dfcolnames.time] - datdf.loc[idx1, dfcolnames.time]

            delta_phi_vec_t = datdf.loc[idx2, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values - datdf.loc[idx1, [dfcolnames.r1_av45, dfcolnames.r2_av45]].values

            a1_t = np.dot(X_vec_t, delta_X_vec_t/delta_T)
            a2_t = np.dot(X_vec_t, delta_phi_vec_t/delta_T)
                    
            # computing constants
            K1 += a1_t**2
            K2 += a2_t**2
            K3 += a1_t*a2_t
            K4 += a1_t*C_t
            K5 += a2_t*C_t
            
    # computing parameters
    alpha1 = (K1*K5 - K3*K4)/(K2*K4 - K3*K5)
    alpha2_gamma = (K3*K3 - K1*K2)/(K2*K4 - K3*K5)
    
    return alpha2_gamma, alpha1


def compute_all_params_woY_bygroup(fulldf, dfcolnames, dtiinfo, grptypdf):
    '''Parameter estimation for a group of subjects together. Alphas, beta,
    and gamma are computed for groups. Time post onset (tpo) is computed
    per subject after beta has been estimated.

    Inputs -
    fulldf: dataframe with subjects, visits, mri, suvr and cognition info
    dfcolnames: class with fields as column names of dataframe
    dtiinfo: class with DTI adjacency and laplacian information
    grptypdf: dataframe with rows as combinations of demographic features that
    describe groups of subjects.

    Outputs-
    outdf: Dataframe with subject ID and the estimated parameters.
    '''
    inv_delta_1_list = []
    delta_2_list = []

    beta_estimated_list = []
    tpo_estimated_list = []
    max_tpo_list = []
    subject_list = []
    
    # list of demographic features used for grouping
    demoglist = grptypdf.columns 
    
    # add columns for new estimates
    newgrptypedf = grptypdf.copy()
    newgrptypedf[['beta_estm','alpha1_estm','alpha2_gamma_estm']] = -1

    for idx,row in grptypdf.iterrows():
        
        print(row.values)
        
        # get subjects with corresponding demographic feature values at baseline
        indexbool = (fulldf[dfcolnames.time]==0)
        
        for ii in range(len(demoglist)):
            indexbool = indexbool & (fulldf[demoglist[ii]]==row[demoglist[ii]])
    
        temp_sub_list = fulldf.loc[indexbool, dfcolnames.subid].unique()

        datdf = fulldf.loc[fulldf[dfcolnames.subid].isin(temp_sub_list)].copy()
        datdf.dropna(subset=[dfcolnames.time, dfcolnames.cogn, 
                             dfcolnames.r1_mri, dfcolnames.r2_mri, 
                             dfcolnames.r1_av45, dfcolnames.r2_av45], inplace=True)

        # estimate alphas and gamma
        alpha2_gamma_estm, alpha1_estm = estimate_alpha_gamma_wo_Y_df(datdf, dfcolnames)

        inv_delta_1_list.append(len(temp_sub_list)*[alpha2_gamma_estm])
        delta_2_list.append(len(temp_sub_list)*[alpha1_estm])

        # estimate beta
        beta_estm = estimate_beta_from_AV45(datdf, dfcolnames, dtiinfo)

        # append to lists
        beta_estimated_list.append(len(temp_sub_list)*[beta_estm])

        # compute best tpo for the given value of beta
        for ridval in temp_sub_list:
            # only give data for this one subject
            subdatdf = datdf.loc[datdf[dfcolnames.subid]==ridval].copy()

            # estimate tpo for the given subject
            tpo_estm, tpolist, objfunclist = find_best_tpo_mse(subdatdf, dfcolnames, dtiinfo, beta_estm)
            tpo_estimated_list.append(tpo_estm)
            max_tpo_list.append(np.floor(subdatdf[dfcolnames.age].min()-50))

        # append subject list
        subject_list.append(temp_sub_list)

        # include estimates in grptypedf
        newgrptypedf.loc[idx,'beta_estm'] = beta_estm
        newgrptypedf.loc[idx,'alpha1_estm'] = alpha1_estm
        newgrptypedf.loc[idx,'alpha2_gamma_estm'] = alpha2_gamma_estm
            
    
    # flatten the lists
    beta_estimated_list = [item2 for item1 in beta_estimated_list for item2 in item1]
    inv_delta_1_list = [item2 for item1 in inv_delta_1_list for item2 in item1]
    delta_2_list = [item2 for item1 in delta_2_list for item2 in item1]
    subject_list = [item2 for item1 in subject_list for item2 in item1]
    
    outdf = pd.DataFrame({dfcolnames.subid:subject_list, 
                          'beta_estm':beta_estimated_list, 
                          'tpo_estm':tpo_estimated_list, 
                          'max_tpo':max_tpo_list, 
                          'alpha1_estm':delta_2_list, 
                          'alpha2_gamma_estm':inv_delta_1_list})

    return outdf, newgrptypedf


def compute_all_params_woY_perpat(fulldf, dfcolnames, dtiinfo):
    '''Parameter estimation for each subject individually. Alphas, beta,
    and gamma are computed for groups. Time post onset (tpo) is computed
    per subject after beta has been estimated.

    Inputs -
    fulldf: dataframe with subjects, visits, mri, suvr and cognition info
    dfcolnames: class with fields as column names of dataframe
    dtiinfo: class with DTI adjacency and laplacian information
 
    Outputs-
    outdf: Dataframe with subject ID and the estimated parameters.
    '''
    inv_delta_1_list = []
    delta_2_list = []

    beta_estimated_list = []
    tpo_estimated_list = []
    max_tpo_list = []

    subject_list = fulldf[dfcolnames.subid].unique()

    for ridval in subject_list:

        print(ridval)
        datadf = fulldf.loc[fulldf[dfcolnames.subid]==ridval].copy()
        datadf.dropna(subset=[dfcolnames.time, dfcolnames.cogn, 
                              dfcolnames.r1_mri, dfcolnames.r2_mri, 
                              dfcolnames.r1_av45, dfcolnames.r2_av45], inplace=True)

        # estimate alphas and gamma
        alpha2_gamma_estm, alpha1_estm = estimate_alpha_gamma_wo_Y_df(datadf, dfcolnames)

        inv_delta_1_list.append(alpha2_gamma_estm)
        delta_2_list.append(alpha1_estm)

        # estimate beta
        beta_estm = estimate_beta_from_AV45(datadf, dfcolnames, dtiinfo)

        # compute best tpo for the given value of beta
        tpo_estm, tpolist, objfunclist = find_best_tpo_mse(datadf, dfcolnames, dtiinfo, beta_estm)

        # append to lists
        beta_estimated_list.append(beta_estm)
        tpo_estimated_list.append(tpo_estm)
        max_tpo_list.append(np.floor(datadf[dfcolnames.age].min()-50))


    outdf = pd.DataFrame({dfcolnames.subid:subject_list, 
                          'beta_estm':beta_estimated_list, 
                          'tpo_estm':tpo_estimated_list, 
                          'max_tpo':max_tpo_list, 
                          'alpha1_estm':delta_2_list, 
                          'alpha2_gamma_estm':inv_delta_1_list})

    return outdf