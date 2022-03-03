import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from scipy import stats
from scipy.stats import ranksums


import param_estimation_v1 as prestm


def load_data(datatype='adni', splitnum=0, sheetname='train'):

	if datatype == 'adni':
		filepath = '../../dataset/processed/'
		filename = 'adni_split%d.xls'%(splitnum)

		df = pd.read_excel(filepath+filename, sheet_name=sheetname)
	
	else:
		print('Wrong data name.')
		return 0
	
	return df


def estimate_params_perpat(df):
	
	# provide field names
	subname = 'RID'
	reg1_av45 = 'HIPPOCAMPAL_SUVR'
	reg2_av45 = 'FRONTAL_SUVR'
	reg1_mri = 'mri_HIPPO_norm'
	reg2_mri = 'mri_FRONT_norm'
	cogvar = 'MMSE_norm'
	agename = 'CurAGE'
	tcname = 'Years'

	# initialize the classes for field names
	dfcolnms = prestm.ColumnNames(subname, tcname, agename, cogvar, 
		reg1_mri, reg2_mri, reg1_av45, reg2_av45)

	admat = np.matrix([[0,1],[1,0]])
	dticlinfo = prestm.DTIMat(admat)

	# estimate parameters per subject
	pmdf = prestm.compute_all_params_woY_perpat(df, dfcolnms, dticlinfo)

	return pmdf


def plot_params(pmdf, pathname):

	parname_list = ['beta_estm','alpha1_estm','alpha2_gamma_estm']
	xlabel_list = [r'$\hat{\beta}$', r'$\hat{\alpha_1}$', r'$\hat{\alpha_{2}\gamma}$']
	xlabel_dict = dict(zip(parname_list, xlabel_list))

	ax = plt.figure(figsize=(12,4))
	for ii in range(len(parname_list)):
	    parname = parname_list[ii]
	    myseries = pmdf[parname]
	    xlabel = xlabel_dict[parname]
	    
	    myseries_adj = myseries[myseries.between(myseries.quantile(.05), myseries.quantile(.95))] 

	    plt.subplot(1,3,ii+1)
	    myseries_adj.hist()
	    plt.xticks(fontsize=14)
	    plt.yticks(fontsize=14)
	    plt.xlabel(xlabel, fontsize=15)
	    
	    if ii==0:
	        plt.ylabel('# individuals', fontsize=15)
	    
	plt.tight_layout()
	plt.savefig(pathname, dpi=300)
	
	return

def param_demog_stats(pmdf, df):
	# include demographic features in df of estimated parameters
	demog_list = ['CurAGE','PTEDUCAT','PTGENDER_num','APOEPOS']
	parname_list = ['beta_estm','alpha1_estm','alpha2_gamma_estm']

	pmdf = pd.concat((pmdf.set_index('RID'), 
	                  df.loc[df.VISCODE=='bl'].set_index('RID')[demog_list]), axis=1)
	pmdf.reset_index(inplace=True)

	# evalute statistical relationship between each demog with each parameter
	dict_parname_list = []
	dict_demog_list = []
	dict_pval_list = []

	for parname in parname_list:
	    
	    for demog_name in demog_list:

	        if demog_name in ['PTEDUCAT','CurAGE']:
	            X = pmdf[demog_name].values
	            y = pmdf[parname].values
	            X2 = sm.add_constant(X)

	            est = sm.OLS(y, X2)
	            est2 = est.fit()

	            assoc_pvalue = est2.pvalues[1]

	        elif demog_name in ['PTGENDER_num', 'APOEPOS']:
	            ignore, assoc_pvalue = ranksums(pmdf.loc[(pmdf[demog_name]==0),parname], 
	                                      pmdf.loc[(pmdf[demog_name]==1),parname])

	        dict_parname_list.append(parname)
	        dict_demog_list.append(demog_name)
	        dict_pval_list.append(assoc_pvalue)
	    
	# store all the resuts in a dataframe
	resdf = pd.DataFrame(dict({'param':dict_parname_list, 
	                                      'demog':dict_demog_list, 
	                                      'pval':dict_pval_list}))

	return resdf

if __name__ == "__main__":

	dataname = 'adni'
	sheet = 'train'
	
	for splitnum in range(1):
		
		# load the data
		df = load_data(dataname, splitnum, sheet)

		# estimate parameters
		pmdf = estimate_params_perpat(df)

		# plot the estimated parameters. MODIFY PATH APPROPRIATELY
		pathname = '../../output/perpat_params_adni_split%d.png'%(splitnum)
		plot_params(pmdf, pathname)

		# perform statistical comparison. MODIFY PATH APPROPRIATELY.
		dfpathname = '../../output/perpat_param_demog_stats_adni_split%d.csv'%(splitnum)
		pvaldf = param_demog_stats(pmdf, df)
		pvaldf.to_csv(dfpathname, index=False)

