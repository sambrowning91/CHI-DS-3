from numpy.random import randint
from scipy.stats import norm
import numpy as np
import scipy
import warnings
import pandas as pd

#####Functions
#confidence_interval(data)
#independent_ttest(groups,outcome)
#paired_ttest(time1,time2)
#anova(groups,outcome)
#partial_correlation(predictor,outcome,covariate)
#correlation(predictor,outcome)
#multipleregression(predictors,outcome)
#logisticregression(predictors,outcome)


class InstabilityWarning(UserWarning):
	"""Issued when results may be unstable."""
	pass

# On import, make sure that InstabilityWarnings are not filtered out.
warnings.simplefilter('always',InstabilityWarning)

def partial_pearson(a,b,c):
	r_ab = scipy.stats.pearsonr(a,b)[0]
	r_ac = scipy.stats.pearsonr(a,c)[0]
	r_bc = scipy.stats.pearsonr(b,c)[0]
	
	r_abc = (r_ab - (r_ac*r_bc)) / np.sqrt( (1-r_ac**2)*(1-r_bc**2))
	return r_abc

def partial_spearman(a,b,c):
	r_ab = scipy.stats.spearmanr(a,b)[0]
	r_ac = scipy.stats.spearmanr(a,c)[0]
	r_bc = scipy.stats.spearmanr(b,c)[0]
	
	r_abc = (r_ab - (r_ac*r_bc)) / np.sqrt( (1-r_ac**2)*(1-r_bc**2))
	return r_abc	

def partial_kendalltau(a,b,c):
	r_ab = scipy.stats.kendalltau(a,b)[0]
	r_ac = scipy.stats.kendalltau(a,c)[0]
	r_bc = scipy.stats.kendalltau(b,c)[0]
	
	r_abc = (r_ab - (r_ac*r_bc)) / np.sqrt( (1-r_ac**2)*(1-r_bc**2))
	return r_abc
	
	
def mediation_pearson(a,b,c):
	r_ab = scipy.stats.pearsonr(a,b)[0]
	r_ac = scipy.stats.pearsonr(a,c)[0]
	r_bc = scipy.stats.pearsonr(b,c)[0]
	
	b = (r_bc - (r_ac*r_ab)) / (1-r_ac**2)
	
	return r_ac*b

def mediation_spearman(a,b,c):
	r_ab = scipy.stats.spearmanr(a,b)[0]
	r_ac = scipy.stats.spearmanr(a,c)[0]
	r_bc = scipy.stats.spearmanr(b,c)[0]
	
	b = (r_bc - (r_ac*r_ab)) / (1-r_ac**2)
	
	return r_ac*b

def mediation_kendalltau(a,b,c):
	r_ab = scipy.stats.kendalltau(a,b)[0]
	r_ac = scipy.stats.kendalltau(a,c)[0]
	r_bc = scipy.stats.kendalltau(b,c)[0]
	
	b = (r_bc - (r_ac*r_ab)) / (1-r_ac**2)
	
	return r_ac*b

def mediation(var1,var2,var3, alpha=0.05, n_samples=10000, type='pearson', epsilon=0.001):
	if type.lower() == "pearson": statfunction = mediation_pearson
	if type.lower() == "kendalltau": statfunction = mediation_kendalltau
	if type.lower() == "spearman": statfunction = mediation_spearman
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	data = (var1,var2,var3)
	# Ensure that the data is actually an array. This isn't nice to pandas,
	#data = tuple( np.array(x) for x in data )

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(var1), n_samples )
	stat = np.array([statfunction(*(x[indexes] for x in data)) for indexes in bootindexes])
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = statfunction(*data)

	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')


	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[nvals]}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]}


def partial_correlation(var1,var2,var3, alpha=0.05, n_samples=10000, type='pearson', epsilon=0.001):
	if type.lower() == "pearson": statfunction = partial_pearson
	if type.lower() == "kendalltau": statfunction = partial_kendalltau
	if type.lower() == "spearman": statfunction = partial_spearman
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	data = (var1,var2,var3)
	# Ensure that the data is actually an array. This isn't nice to pandas,
	#data = tuple( np.array(x) for x in data )

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(var1), n_samples )
	stat = np.array([statfunction(*(x[indexes] for x in data)) for indexes in bootindexes])
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = statfunction(*data)

	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')


	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[nvals]}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]}

	
def anova_f(groups,outcome):
	categories = sorted(groups.unique())
	data = [outcome[(groups == category)] for category in categories]
	fv,p = scipy.stats.f_oneway(*data)
	return fv

	
def student_t(groups,outcome):
	categories = groups.unique()
	data = [outcome[(groups == category)] for category in categories]
	data.append(equal_var=False)
	tv,p = scipy.stats.ttest_ind(*data)
	return tv
	
def ttest_diff(groups,outcome):
	categories = sorted(groups.unique())
	return np.mean(outcome[(groups == categories[0])]) - np.mean(outcome[(groups == categories[1])])

def anova(groups,outcome,alpha=0.05, n_samples=10000, epsilon=0.001):
	statfunction = anova_f
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	data = (groups,outcome)

	bootindexes = bootstrap_indexes(len(groups), n_samples )
	stat = np.array([np.mean(statfunction(*(x[indexes] for x in data))) for indexes in bootindexes])
	
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = np.mean(statfunction(*data))
	
	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [np.mean(statfunction(*(x[indexes] for x in data))) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')


	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[nvals]}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]}

def difference_paired_means(var1,var2,alpha=0.05, n_samples=10000, epsilon=0.001):
	data = var1-var2
	return confidence_interval(data,alpha, n_samples, epsilon=0.001)

def difference_independent_means(var1,var2,alpha=0.05, n_samples=1000, epsilon=0.001):
	statfunction = ttest_diff
	
	#if isinstance(var1, pd.Series) == False: var1=pd.Series(var1)
	#if isinstance(var2, pd.Series) == False: var2=pd.Series(var2)
	
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	print var2
	data = (var1, var2)
	# Ensure that the data is actually an array. This isn't nice to pandas,
	#data = tuple( np.array(x) for x in data )

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(var1), n_samples )
	stat = np.array([np.mean(statfunction(*(x[indexes] for x in data))) for indexes in bootindexes])
	
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = np.mean(statfunction(*data))
	
	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [np.mean(statfunction(*(x[indexes] for x in data))) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')


	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[nvals]}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]}

	
def confidence_interval(data, alpha=0.05, n_samples=10000, epsilon=0.001):
	statfunction=np.average
	data = np.array(data)
	data = (data,)
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(data[0]), n_samples )
	stat = np.array([statfunction(*(x[indexes] for x in data)) for indexes in bootindexes])
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method
	# The value of the statistic function applied just to the actual data.
	ostat = statfunction(*data)

	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )
	
	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')


	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(data),"%.2f%% Confidence Interval" % round(1- alpha,2): stat[nvals]}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(data),"%.2f%% Confidence Interval" % round(1- alpha,2) : stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]}

from sklearn import linear_model
def multipleregression(pred,y):
	clf = linear_model.LinearRegression().fit(pred,y)
	coefs = [clf.intercept_]
	coefs.extend(clf.coef_)
	return  coefs
	
def logisticregression(pred,y):
	clf = linear_model.LogisticRegression().fit(pred,y)
	coefs = [clf.intercept_]
	coefs.extend(clf.coef_)
	return  coefs

def logistic_regression(predictors,outcome, alpha=0.05, n_samples=10000, epsilon=0.001):
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	data = (predictors,outcome)
	statfunction = logisticregression
	# Ensure that the data is actually an array. This isn't nice to pandas,
	#data = tuple( np.array(x) for x in data )

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(outcome), n_samples )
	stat = np.array([statfunction(*(x[indexes] for x in data)) for indexes in bootindexes])
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = statfunction(*data)

	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')

	from itertools import izip
	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2) :  zip(*stat[nvals])}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(*data),"%.2f %% Confidence Interval" % (1- alpha) : zip(*stat[(nvals, np.indices(nvals.shape)[1:].squeeze())])}
		
			
	
def linear_regression(predictors,outcome, alpha=0.05, n_samples=10000, epsilon=0.001):
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	data = (predictors,outcome)
	statfunction = regression
	# Ensure that the data is actually an array. This isn't nice to pandas,
	#data = tuple( np.array(x) for x in data )

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(outcome), n_samples )
	stat = np.array([statfunction(*(x[indexes] for x in data)) for indexes in bootindexes])
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = statfunction(*data)

	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')

	from itertools import izip
	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data),"%.2f%% Confidence Interval" % round(1- alpha,2) :  zip(*stat[nvals])}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(*data),"%.2f %% Confidence Interval" % (1- alpha) : zip(*stat[(nvals, np.indices(nvals.shape)[1:].squeeze())])}
		
		
def correlation(var1,var2, alpha=0.05, n_samples=10000, type='pearson', epsilon=0.001):
	if type.lower() == "pearson": statfunction = scipy.stats.pearsonr
	if type.lower() == "kendalltau": statfunction = scipy.stats.kendalltau
	if type.lower() == "spearman": statfunction = scipy.stats.spearmanr
	# Deal with the alpha values
	if np.iterable(alpha):
		alphas = np.array(alpha)
	else:
		alphas = np.array([alpha/2,1-alpha/2])
	data = (var1,var2)
	# Ensure that the data is actually an array. This isn't nice to pandas,
	#data = tuple( np.array(x) for x in data )

	# We don't need to generate actual samples; that would take more memory.
	# Instead, we can generate just the indexes, and then apply the statfun
	# to those indexes.
	bootindexes = bootstrap_indexes(len(var1), n_samples )
	stat = np.array([statfunction(*(x[indexes] for x in data))[0] for indexes in bootindexes])
	stat.sort(axis=0)

	# Bias-Corrected Accelerated Method


	# The value of the statistic function applied just to the actual data.
	ostat = statfunction(*data)[0]

	# The bias correction value.
	z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

	# Statistics of the jackknife distribution
	jackindexes = jackknife_indexes(data[0])
	jstat = [statfunction(*(x[indexes] for x in data))[0] for indexes in jackindexes]
	jmean = np.mean(jstat,axis=0)

	# Acceleration value
	a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

	zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

	avals = norm.cdf(z0 + zs/(1-a*zs))


	nvals = np.round((n_samples-1)*avals)
	
	if np.any(nvals==0) or np.any(nvals==n_samples-1):
		warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
	elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
		warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

	nvals = np.nan_to_num(nvals).astype('int')


	if nvals.ndim == 1:
		# All nvals are the same. Simple broadcasting
		return {"Estimate":statfunction(*data)[0],"%.2f%% Confidence Interval" % round(1- alpha,2) : stat[nvals]}
	else:
		# Nvals are different for each data point. Not simple broadcasting.
		# Each set of nvals along axis 0 corresponds to the data at the same
		# point in other axes.
		return {"Estimate":statfunction(*data)[0],"%.2f %% Confidence Interval" % (1- alpha) : stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]}



def bootstrap_indexes(rows, n_samples=10000):
	"""
Given data points data, where axis 0 is considered to delineate points, return
an generator for sets of bootstrap indexes. This can be used as a list
of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
	"""
	for _ in xrange(n_samples):
		yield randint(rows, size=(rows,))

def jackknife_indexes(data):
	"""
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.

For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.
	"""
	base = np.arange(0,len(data))
	return (np.delete(base,i) for i in base)

def subsample_indexes(data, n_samples=1000, size=0.5):
	"""
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is indexes a subsample of the data of size
``size``. If size is >= 1, then it will be taken to be an absolute size. If
size < 1, it will be taken to be a fraction of the data size. If size == -1, it
will be taken to mean subsamples the same size as the sample (ie, permuted
samples)
	"""
	if size == -1:
		size = len(data)
	elif (size < 1) and (size > 0):
		size = round(size*len(data))
	elif size > 1:
		pass
	else:
		raise ValueError("size cannot be {0}".format(size))
	base = np.tile(np.arange(len(data)),(n_samples,1))
	for sample in base: np.random.shuffle(sample)
	return base[:,0:size]

