"""
Here there be tests.
"""

import json
import sys
import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
from glob import glob
import os
import seaborn as sns
from tqdm import tqdm
import time 
import matplotlib as mpl
import matplotlib.pyplot as plt
from simulate_transit import * 
from simulate_helpers import *

#input_path = '/blue/sarahballard/c.lam/sculpting2/' # HPG
#output_path = '/blue/sarahballard/c.lam/sculpting2/mastrangelo/' # HPG
path = '/Users/chris/Desktop/mastrangelo/' # new computer has different username
berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
#berger_kepler = pd.read_csv(input_path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# make berger_kepler more wieldy
berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
						     'iso_age', 'iso_age_err1', 'iso_age_err2', 'iso_mass', 'iso_rad', 'rrmscdpp06p0',
							 'fractional_err1', 'logR', 'fractional_err2', 'is_giant']]

k = pd.Series([833, 134, 38, 15, 5, 0])
G = 6.6743e-8 # gravitational constant in cgs

def prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c):
	"""
	Each model run will use an evenly spaced (m,b, cutoff) tuple on a discrete 11x11x11 3D grid
	We're doing log(time), so slope is sampled linearly (everything gets logged together later)
	If a cutoff results in a zero probability, don't bother 

	gi_m: grid index on m axis
	gi_b: grid index on b axis
	gi_c: grid index for cutoff time axis
	"""
	#cube[0] = -1e-9*np.logspace(8,10,11)[gi_m] # convert from year to Gyr
	cube[0] = np.linspace(-1,0,6)[gi_m] 
	cube[1] = np.linspace(0,1,11)[gi_b]
	#cube[2] = np.logspace(1e8,1e10,11)
	cube[2] = np.logspace(8,10,11)[gi_c] # in Ballard et al in prep, they use log(yrs) instead of drawing yrs from logspace
	return cube

def better_loglike(lam, k):
	"""
	Calculate Poisson log likelihood
	Changed 0 handling from simulate.py to reflect https://www.aanda.org/articles/aa/pdf/2009/16/aa8472-07.pdf

	Params: 
	- lam: model predictions for transit multiplicity (list of ints)
	- k: Kepler transit multiplicity (list of ints); can accept alternate ground truths as well

	Returns: Poisson log likelihood (float)
	"""

	logL = []
	#print(lam)
	for i in range(len(lam)):
		if lam[i]==0:
			term3 = -lgamma(k[i]+1)
			term2 = -lam[i]
			term1 = 0
			logL.append(term1+term2+term3)

		else:
			term3 = -lgamma(k[i]+1)
			term2 = -lam[i]
			term1 = k[i]*np.log(lam[i])
			logL.append(term1+term2+term3)

	return np.sum(logL)

# how many params, how many dims, initialize cube
ndim = 3
nparams = 3
cube = [0, 0, 0]

###################
## TESTS BEGIN HERE
###################

def test1(gi_m, gi_b, gi_c, f, cube):

	"""
	Test 1: Unit tests to see how much information a sculpting law contains
	"""

	cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
	print("cube: ", cube)

	berger_kepler_planets = model_vectorized(berger_kepler, 'limbach-hybrid', cube)
	berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
			'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
			'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
			'prob_detections','sn']] 
	
	# isolate transiting planets
	transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]

	#print("Hmm: ", berger_kepler_planets.iso_age, berger_kepler_planets.prob_intact)

	# compute transit multiplicity 
	transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity = transit_multiplicity.to_list()
	print(transit_multiplicity)

	return cube, berger_kepler_planets


#_, bkp1 = test1(1, 1, 7, 0.3, cube)
#cube, bkp2 = test1(1, 1, 1, 0.3, cube)
#test1(1, 2, 7, 0.3, cube)
#test1(1, 3, 7, 0.3, cube)
#test1(3, 1, 1, 0.3, cube) # no dupe perturb
_, bkp2 = test1(3, 1, 10, 0.3, cube)
_, bkp3 = test1(0, 1, 10, 0.3, cube)

x = bkp2.iso_age
#x = np.ones(len(x)) * 2.
m = cube[0]
b = cube[1]
cutoff = cube[2]
print("cutoff: ", cutoff)
print("ages: ", x)

print(np.where(((x*1e9 > 1e8) & (x*1e9 <= cutoff)), b+m*(np.log10(x*1e9)-8), np.where(
                    x*1e9 > cutoff, b+m*(np.log10(cutoff)-8), b)))
print("what it should be: ", b+m*(np.log10(cutoff)-8))

"""
print(bkp1.loc[bkp1.prob_intact > 0].drop_duplicates(subset=['kepid']))
plt.hist(bkp1.drop_duplicates(subset=['kepid']).prob_intact)
plt.savefig(path+'test1.png')
plt.show()

"""
print(bkp2.loc[bkp2.prob_intact > 0].drop_duplicates(subset=['kepid']))
plt.hist(bkp2.drop_duplicates(subset=['kepid']).prob_intact)
plt.savefig(path+'test2.png')
plt.show()

print(bkp3.loc[bkp3.prob_intact > 0].drop_duplicates(subset=['kepid']))
plt.hist(bkp3.drop_duplicates(subset=['kepid']).prob_intact)
plt.savefig(path+'test3.png')
plt.show()


def test2():

	"""
	Test 2: Do different sculpting laws produce reasonably different transit multiplicity yields?
	"""

	### Scenario 1: fast sculpting
	# fetch hyperparams
	gi_m = 0
	gi_b = 5
	gi_c = 8
	cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
	f = 0.3
	print("cube: ", cube)

	berger_kepler_planets = model_vectorized(berger_kepler, 'limbach-hybrid', cube)
	berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
			'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
			'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
			'prob_detections','sn']] 

	# isolate transiting planets
	transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]

	print("Hmm: ", berger_kepler_planets.iso_age, berger_kepler_planets.prob_intact)

	# compute transit multiplicity 
	transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity = transit_multiplicity.to_list()
	print(transit_multiplicity)
	intrinsic_intact = berger_kepler_planets.loc[berger_kepler_planets.prob_intact==1]
	intrinsic_disrupted = berger_kepler_planets.loc[berger_kepler_planets.prob_intact==0]
	print("intacts: ", len(intrinsic_intact.kepid.unique()))
	print("disrupteds: ", len(intrinsic_disrupted.kepid.unique()))
	print("")
	plt.hist(berger_kepler_planets.drop_duplicates(subset=['kepid']).prob_intact)
	plt.savefig(path+'test1.png')

	### Scenario 2: medium sculpting
	# fetch hyperparams
	gi_m = 5
	gi_b = 5
	gi_c = 8
	cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
	f = 0.3
	print("cube: ", cube)

	berger_kepler_planets = model_vectorized(berger_kepler, 'limbach-hybrid', cube)
	berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
			'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
			'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
			'prob_detections','sn']] 

	# isolate transiting planets
	transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]

	print("Hmm: ", berger_kepler_planets.iso_age, berger_kepler_planets.prob_intact)

	# compute transit multiplicity 
	transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity = transit_multiplicity.to_list()
	print(transit_multiplicity)

	intrinsic_intact = berger_kepler_planets.loc[berger_kepler_planets.prob_intact==1]
	intrinsic_disrupted = berger_kepler_planets.loc[berger_kepler_planets.prob_intact==0]
	print("intacts: ", len(intrinsic_intact.kepid.unique()))
	print("disrupteds: ", len(intrinsic_disrupted.kepid.unique()))
	print("")
	plt.hist(berger_kepler_planets.drop_duplicates(subset=['kepid']).prob_intact)
	plt.savefig(path+'test2.png')

	### Scenario 3: no sculpting
	# fetch hyperparams
	gi_m = 10
	gi_b = 5
	gi_c = 8
	cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
	f = 0.3
	print("cube: ", cube)

	berger_kepler_planets = model_vectorized(berger_kepler, 'limbach-hybrid', cube)
	berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
			'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
			'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
			'prob_detections','sn']] 

	# isolate transiting planets
	transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]

	# compute transit multiplicity 
	transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity = transit_multiplicity.to_list()
	print(transit_multiplicity)

	#print(berger_kepler_planets)
	print("Hmm: ", berger_kepler_planets.iso_age, berger_kepler_planets.prob_intact)
	#print(berger_kepler_planets.prob_intact)
	# check that intact probability calculator is okay...it's not
	intrinsic_intact = berger_kepler_planets.loc[berger_kepler_planets.prob_intact==1]
	intrinsic_disrupted = berger_kepler_planets.loc[berger_kepler_planets.prob_intact==0]
	print("intacts: ", len(intrinsic_intact.kepid.unique()))
	print("disrupteds: ", len(intrinsic_disrupted.kepid.unique()))
	print("")
	plt.hist(berger_kepler_planets.drop_duplicates(subset=['kepid']).prob_intact)
	plt.savefig(path+'test3.png')