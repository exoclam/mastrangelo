"""
This is a smaller version of simulate_main.py, in order to create a subsample of the original fake planetary system dataset
after the great Water Bottle Disaster of March 2023 (the water bottle didn't kill the data, but it sounds more dramatic this way).
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

input_path = '/blue/sarahballard/c.lam/sculpting2/' # HPG
output_path = '/blue/sarahballard/c.lam/sculpting2/mastrangelo/' # HPG
#path = '/Users/chris/Desktop/mastrangelo/' # new computer has different username
#berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
berger_kepler = pd.read_csv(input_path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# make berger_kepler more wieldy
berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
						     'iso_age', 'iso_age_err1', 'iso_age_err2', 'iso_mass', 'iso_rad', 'rrmscdpp06p0',
							 'fractional_err1', 'logR', 'fractional_err2', 'is_giant']]

#pnum = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv') # planet hosts among crossmatched Berger sample
#k = pnum.groupby('kepid').count().koi_count.reset_index().groupby('koi_count').count()
k = pd.Series([833, 134, 38, 15, 5, 0])
G = 6.6743e-8 # gravitational constant in cgs

# how many params, how many dims, initialize cube
ndim = 3
nparams = 3
cube = [0, 0, 0]

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
	cube[0] = np.linspace(-2,0,11)[gi_m] 
	cube[1] = np.linspace(0,1,3)[gi_b]
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


def main_ground_truth(cube, ndim, nparams):
	"""
	CREATE FAKE GROUND TRUTH DATASET BY SIMULATING SYSTEMS FOR EACH STAR USING THE GIVEN MEAN STELLAR VALUES FROM KEPLER-GAIA CROSSMATCH
	For each of the three main hyperparams, I make N simulations (N * 10000 total simulations). 
	I output each simulation to a folder and never run them again. The I/O for calculating logLs will be worth not having to re-run simulations.
	For each of the N * 10K resulting lambdas, I create 10 of varying fraction of systems with planets (the fourth hyperparam).
	I run separate code to compute logLs for each 4-hyperparam combination and plot that as I have done before.
	I do the same, now using the Rayleigh-Limbach hybrid eccentricity distribution. 

	Params: 
	- cube: [m, b, cutoff]
	- ndim: number of dimensions, will be 4 instead of 3 for pymultinest
	- nparams: number of parameters, will be 4 instead of 3 for pymultinest
	"""

	### ad hoc logic bc HPG ran out of memory and I don't want to redo already-finished simulations
	done = glob(path+'systems/transits*')

	cube = prior_grid_logslope(cube, ndim, nparams, 0, 0, 0)

	"""
	### trivial case of no sculpting
	output_filename = path + 'systems/transits0_0_0_0.csv'
	if output_filename not in done:
		berger_kepler_planets = model_vectorized(berger_kepler, 'limbach-hybrid', cube)
		berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
				'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
				'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
				'prob_detections','sn']]
		
		berger_kepler_planets.to_csv(output_filename, sep='\t')
	
	else:
		pass
	"""

	# other models
	for gi_m in range(11):
		for gi_b in range(2):

			# increment to account for trivial case already being run
			gi_b = gi_b + 1 

			for gi_c in range(11): # NEED TO RUN 4, 5, 6, 7, 9, 10, 11...or, you know, just do a done-ness check

				# fetch hyperparams
				cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)

				# generate filename
				output_filename = output_path + 'systems/transits' +str(gi_m) + '_' + str(gi_b) + '_' + str(gi_c) + '.csv'

				if output_filename not in done:
					berger_kepler_planets = model_vectorized(berger_kepler, 'limbach-hybrid', cube)
					berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
							'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
							'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
							'prob_detections','sn']]
			
					berger_kepler_planets.to_csv(output_filename, sep='\t')

				else:
					pass

	return


def main_recovery(cube, ndim, nparams):
	"""
	CREATE 30 REALIZATIONS FOR EACH STAR, USING ERRORS.
	FOR EACH REALIZATION, COMPARE 
	"""

	done = glob(path+'systems-recovery/transits*')

	# do the trivial case of everybody is disrupted, just once
	#cube = prior_grid_logslope(cube, ndim, nparams, 0, 0, 0)
	"""
	for i in range(30):
		berger_kepler_temp = draw_star(berger_kepler)
		#output_filename = output_path + 'systems-recovery/transits0_0_0_' + str(i) + '.csv'
		output_filename = path + 'systems-recovery/transits0_0_0_' + str(i) + '.csv'
		if output_filename not in done:
			berger_kepler_planets = model_vectorized(berger_kepler_temp, 'limbach-hybrid', cube)
			berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
					'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
					'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
					'prob_detections','sn']]
			
			geom_sum = len(berger_kepler_planets.loc[berger_kepler_planets.geom_transit_status == 1])
			detected_sum = len(berger_kepler_planets.loc[berger_kepler_planets.transit_status == 1])

			berger_kepler_planets.to_csv(output_filename)
		else:
			pass
	"""

	# now do the rest
	for gi_m in range(11):
		for gi_b in range(2):

			# increment to account for trivial case already being run
			gi_b = gi_b + 1 
				
			for gi_c in range(11): 

				# fetch hyperparams
				cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)

				for i in range(30):
					berger_kepler_temp = draw_star(berger_kepler)
					#output_filename = output_path + 'systems-recovery/transits' +str(gi_m) + '_' + str(gi_b) + '_' + str(gi_c) + '_' + str(i) + '.csv'
					output_filename = output_path + 'systems-recovery/transits' +str(gi_m) + '_' + str(gi_b) + '_' + str(gi_c) + '_' + str(i) + '.csv'
					if output_filename not in done:

						berger_kepler_planets = model_vectorized(berger_kepler_temp, 'limbach-hybrid', cube)
						berger_kepler_planets = berger_kepler_planets[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
								'iso_age', 'iso_age_err1', 'iso_age_err2', 'logR','is_giant','fractional_err1','fractional_err2','prob_intact','midplanes',
								'intact_flag','sigma','num_planets','P','incl','mutual_incl','ecc','omega','lambda_ks','second_terms','geom_transit_status','transit_status',
								'prob_detections','sn']]
						berger_kepler_planets.to_csv(output_filename)

					else:
						pass

	return


"""
VECTORIZATION SPEED TEST
cube = prior_grid_logslope(cube, ndim, nparams, 0, 0, 0)

start = time.time()
model_vectorized(berger_kepler, 'limbach-hybrid', cube)
end = time.time()
print("elapsed vectorized: ", end-start)
# it was 32 seconds

start = time.time()
model_van_eylen(berger_kepler.iso_age, berger_kepler, 'limbach-hybrid', cube)
end = time.time()
print("elapsed non-vectorized: ", end-start)
# it was 171 seconds, or about 6 times slower 
"""

main_ground_truth(cube, ndim, nparams)
