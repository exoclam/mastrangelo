### Collect the outputs of injection_recovery_main.py

import json
import sys
import os
from glob import glob
import numpy as np
import pandas as pd
from math import lgamma
#from simulate_main import prior_grid_logslope, better_loglike
from datetime import datetime
from itertools import zip_longest
import numpy.ma as ma # for masked arrays

input_path = '/blue/sarahballard/c.lam/sculpting2/' # HPG
output_path = '/blue/sarahballard/c.lam/sculpting2/mastrangelo/' # HPG
path = '/Users/chris/Desktop/mastrangelo/' # new computer has different username
#berger_kepler = pd.read_csv(input_path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# make berger_kepler more wieldy
#berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
#						     'iso_age', 'iso_age_err1', 'iso_age_err2']]

# ground truth from Kepler observed transit multiplicity
#pnum = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv') # planet hosts among crossmatched Berger sample
#k = pnum.groupby('kepid').count().koi_count.reset_index().groupby('koi_count').count()
k = pd.Series([833, 134, 38, 15, 5, 0])
G = 6.6743e-8 # gravitational constant in cgs

# set up hypercube just so I can associate logLs with correct hyperparams
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

	# in the rare occasion that a simulated system has 7+ planets, throw them into the 6+ bin
	if len(k) < len(lam): 
		extras = lam[len(k):]
		sum_extras = np.sum(extras)
		lam[5] += sum_extras
	lam = lam[:6]

	logL = []
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


def normal_loglike(x, mu, sigma):
	"""
	Calculate Gaussian log likelihood
	lnp(x) = -0.5 * ((x-mu)/sigma)**2 - ln(sigma * sqrt(2*pi))
	Inputs: 
	- x: data
	- mu: mean of the random variable
	- sigma: standard deviation of the random variable

	Returns: Normal log likelihood (float)
	"""

	term1 = -0.5 * ((x-mu)/sigma)**2
	term2 = np.log(sigma * np.sqrt(2*np.pi))
	logL = term1 - term2

	return logL


def collect(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs):
	
	"""
	Compute geometric and detected transit multiplicities, intact/disrupted fractions, and log likelihood.

	Inputs: 
	- df: read-in DataFrames of the simulated planetary system products of injection_recovery_main.py
	- f: fraction of planet-hosting stars
	- transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs: future columns in CSV output of collect_simulations.py

	Outputs:
	- transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs: see description in Inputs

	"""
	
	# isolate transiting planets
	transiters_berger_kepler = df.loc[df['transit_status']==1]

	# compute transit multiplicity 
	transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity = transit_multiplicity.to_list()
	try:
		transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
	except:
		print(transit_multiplicity.to_list())

	transit_multiplicities.append(transit_multiplicity)

	# also calculate the geometric transit multiplicity
	geom_transiters_berger_kepler = df.loc[df['geom_transit_status']==1]
	geom_transit_multiplicity = f * geom_transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	geom_transit_multiplicity = geom_transit_multiplicity.to_list()
	geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k
	geom_transit_multiplicities.append(geom_transit_multiplicity)

	# calculate logLs 
	logL = better_loglike(transit_multiplicity, k)
	logLs.append(logL)

	# get intact and disrupted fractions (combine them later to get fraction of systems w/o planets)
	intact = df.loc[df.intact_flag=='intact']
	disrupted = df.loc[df.intact_flag=='disrupted']
	intact_frac = f*len(intact)/len(df)
	disrupted_frac = f*len(disrupted)/len(df)
	intact_fracs.append(intact_frac)
	disrupted_fracs.append(disrupted_frac)

	return transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs


def collect_past_ii(df, f):
	
	"""
	Collect age spread across different transit multiplicity bins for each model. 
	Transit multiplicities are capped at 3+, and, once relevant, logLs will be calculated to fit to the age of a system as a function of multiplicity.

	Inputs: 
	- df: read-in DataFrames of the simulated planetary system products of injection_recovery_main.py
	- f: fraction of planet-hosting stars

	Outputs:
	- age means and standard deviations for each multiplicity bin

	"""

	# isolate non-transiting systems
	nontransit = df.loc[df['transit_status']==0]

	# isolate transiting systems
	transit = df.loc[df['transit_status']==1]

	# if f < 1, randomly steal some non-nontransits for the nontransits gang
	samples_indices = transit.sample(frac=f, replace=False).index 
	transit.loc[samples_indices, 'transit_status'] = 0

	# rejects
	rejects = transit.loc[transit['transit_status'] == 0]

	# concatenate rejects to nontransit DataFrame
	nontransit = pd.concat([nontransit, rejects])

	# new transit DataFrame no longer has rejects
	transit = transit.loc[transit['transit_status'] != 0]

	# get age spread for non-transiting systems
	zeros_age_mean = np.mean(nontransit.iso_age.astype(float))
	zeros_age_std = np.mean(0.5 * (np.array(nontransit.iso_age_err1.astype(float)) + np.abs(np.array(nontransit.iso_age_err2.astype(float)))))

	# create column counting number of planets per system
	transit['yield'] = 1
	transit = transit.groupby(['kepid', 'iso_age', 'iso_age_err1', 'iso_age_err2'])[['yield']].count().reset_index()

	# get age spread for each count
	ones = transit.loc[transit['yield'] == 1]
	ones_age_mean = np.mean(ones.iso_age.astype(float))
	ones_age_std = np.mean(0.5 * (np.array(ones.iso_age_err1.astype(float)) + np.abs(np.array(ones.iso_age_err2.astype(float)))))

	twos = transit.loc[transit['yield'] == 2]
	twos_age_mean = np.mean(twos.iso_age.astype(float))
	twos_age_std = np.mean(0.5 * (np.array(twos.iso_age_err1.astype(float)) + np.abs(np.array(twos.iso_age_err2.astype(float)))))

	threes = transit.loc[transit['yield'] >= 3]
	threes_age_mean = np.mean(threes.iso_age.astype(float))
	threes_age_std = np.mean(0.5 * (np.array(threes.iso_age_err1.astype(float)) + np.abs(np.array(threes.iso_age_err2.astype(float)))))

	return zeros_age_mean, zeros_age_std, ones_age_mean, ones_age_std, twos_age_mean, twos_age_std, threes_age_mean, threes_age_std


sims = []
ms = []
bs = []
cs = []
fs = []

start = datetime.now()
#print("start: ", start)

#sim = glob(output_path+'systems-recovery-redo/transits0_0_0_0.csv')
cube = prior_grid_logslope(cube, ndim, nparams, 0, 0, 0)
transit_multiplicities = []
geom_transit_multiplicities = []
intact_fracs = []
disrupted_fracs = []
logLs = []
"""
nontransit_age_maxes = []
nontransit_age_mins = []
ones_age_maxes = []
ones_age_mins = []
twos_age_maxes = []
twos_age_mins = []
threes_age_maxes = []
threes_age_mins = []
"""
#output = pd.DataFrame()

""""
gi_m = 2
gi_b = 2
gi_c = 1
sim = glob(output_path+'systems-recovery-redo/transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'*')
print(sim)
cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
print(gi_m, gi_b, gi_c, len(sim)) # so I know where I am

# cycle through different fractions of systems with planets
for f in np.round(np.linspace(0.1, 1, 10), 1):
	
	for i in range(len(sim)):
		#print("hi: ", gi_m, gi_b, gi_c, cube[0], cube[1], cube[2])
		ms.append(cube[0])
		bs.append(cube[1])
		cs.append(cube[2])
		fs.append(f)

		df = pd.read_csv(sim[i], sep=',', on_bad_lines='skip')
		print(i, df)
		# populate future columns for output DataFrame
		transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs = collect(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs)

quit()
"""

# cycle through different fractions of systems with planets
"""
for f in np.linspace(0.1, 1, 10):

	for i in range(len(sim)):

		ms.append(cube[0])
		bs.append(cube[1])
		cs.append(cube[2])
		fs.append(f)
					
		df = pd.read_csv(sim[i], sep=',', on_bad_lines='skip')

		# populate future columns for output DataFrame
		transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs = collect(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs)
		#nontransit_age_max, nontransit_age_min, ones_age_max, ones_age_min, twos_age_max, twos_age_min, threes_age_max, threes_age_min = collect_past_ii(df, f)
		#nontransit_age_maxes.append(nontransit_age_max)
		#nontransit_age_mins.append(nontransit_age_min)
		#ones_age_maxes.append(ones_age_max)
		#ones_age_mins.append(ones_age_min)
		#twos_age_maxes.append(twos_age_max)
		#twos_age_mins.append(twos_age_min)
		#threes_age_maxes.append(threes_age_max)
		#threes_age_mins.append(threes_age_min)
"""
print("finished initial set")

for gi_m in range(6):

	for gi_b in range(11):
		
		for gi_c in range(11):
			
			try:
				sim = glob(output_path+'systems-recovery-loguniform-redo/transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'*')
				#sim = glob(output_path+'systems-ten/transits'+str(gi_m)+'_'+str(gi_b)+'_1*')
			except:
				print("file not found: ", 'transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'*')
				#print("file not found: ", 'transits'+str(gi_m)+'_'+str(gi_b)+'_1*')
				continue # if no file found, skip to next iteration 

			cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
			print(gi_m, gi_b, gi_c, len(sim)) # so I know where I am

			# cycle through different fractions of systems with planets
			for f in np.round(np.linspace(0.1, 1, 10), 1):
				
				for i in range(len(sim)):

					df = pd.read_csv(sim[i], sep='\t', on_bad_lines='skip') # ground truth is \t; recovery is ,

					# populate future columns for output DataFrame
					try: # Some sim dfs got corrupted in HPG. We skip these. 
						transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs = collect(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs)
						#nontransit_age_max, nontransit_age_min, ones_age_max, ones_age_min, twos_age_max, twos_age_min, threes_age_max, threes_age_min = collect_past_ii(df, f)
						ms.append(cube[0])
						bs.append(cube[1])
						cs.append(cube[2])
						fs.append(f)

					except:
						continue
					
					"""
					nontransit_age_maxes.append(nontransit_age_max)
					nontransit_age_mins.append(nontransit_age_min)
					ones_age_maxes.append(ones_age_max)
					ones_age_mins.append(ones_age_min)
					twos_age_maxes.append(twos_age_max)
					twos_age_mins.append(twos_age_min)
					threes_age_maxes.append(threes_age_max)
					threes_age_mins.append(threes_age_min)
					"""

#end = datetime.now()
#print("TIME ELAPSED: ", end-start)

df_logL = pd.DataFrame({'ms': ms, 'bs': bs, 'cs': cs, 'fs': fs, 'transit_multiplicities': transit_multiplicities, 
			'geom_transit_multiplicities': geom_transit_multiplicities, 'intact_fracs': intact_fracs, 'disrupted_fracs': disrupted_fracs, 'logLs': logLs})
			
	#'nontransit_age_maxes': nontransit_age_maxes, 'nontransit_age_mins': nontransit_age_mins, 'ones_age_maxes': ones_age_maxes, 
	#'ones_age_mins': ones_age_mins, 'twos_age_maxes': twos_age_maxes, 'twos_age_mins': twos_age_mins,
	#'threes_age_maxes': threes_age_maxes, 'threes_age_mins': threes_age_mins})
print(df_logL)

df_logL.to_csv(output_path+'collect_ground_truth_loguniform_redo.csv', index=False) # collect_ is for transit multiplicity; past_ii_ is for age vs multiplicity

quit()

# calculate max/min envelopes for both multiplicities
lam_elt_max = []
lam_elt_min = []
lam_elt_avg = []

for temp_list in zip_longest(*transit_multiplicities_all):
	elt = [0 if v is None else v for v in ma.masked_values(temp_list, 0)]
	print(elt)
	lam_elt_max.append(max(elt))
	lam_elt_min.append(min(elt))
	lam_elt_avg.append(np.mean(elt))

lam_upper.append(lam_elt_max)
lam_lower.append(lam_elt_min)
lam_avgs.append(lam_elt_avg)
print(lam_upper, lam_lower, lam_avgs)				
fasdfa

"""
print(len(ms))
print(len(bs))
print(len(cs))
print(len(fs))
print(len(transit_multiplicities_all))
"""

df_logL = pd.DataFrame({'ms': ms, 'bs': bs, 'cs': cs, 'fs': fs, 'logLs': logLs, 
	'transit_multiplicities': transit_multiplicities_all, 'geom_transit_multiplicities': geom_transit_multiplicities_all,
	'intact_fracs': intact_fracs_all, 'disrupted_fracs': disrupted_fracs_all})
print(df_logL)
df_logL.to_csv(path+'logLs.csv', index=False)
