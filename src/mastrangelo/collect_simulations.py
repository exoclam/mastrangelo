# for each {m, b, cutoff} filename, read all 9 files
# calculate logLs and get mean and std (or min and max)
# read out to plot in Jupyter locally

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
	cube[0] = np.linspace(-2,0,3)[gi_m] 
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

	# compute transit multiplicity and save off the original transit multiplicity (pre-frac)
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


def collect2(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs):
	
	"""
	Compute geometric and detected transit multiplicities, intact/disrupted fractions, and log likelihood.
	However, transit multiplicities are capped at 3+, and logLs are calculated to fit to the age of a system as a function of multiplicity.

	Inputs: 
	- df: read-in DataFrames of the simulated planetary system products of injection_recovery_main.py
	- f: fraction of planet-hosting stars
	- transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs: future columns in CSV output of collect_simulations.py

	Outputs:
	- transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs: see description in Inputs

	"""
	
	# isolate transiting planets
	transiters_berger_kepler = df.loc[df['transit_status']==1]

	# compute transit multiplicity and save off the original transit multiplicity (pre-frac)
	transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
	transit_multiplicities.append(list(transit_multiplicity))

	# also calculate the geometric transit multiplicity
	geom_transiters_berger_kepler = df.loc[df['geom_transit_status']==1]
	geom_transit_multiplicity = f * geom_transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k
	geom_transit_multiplicities.append(list(geom_transit_multiplicity))

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


sims = []
ms = []
bs = []
cs = []
fs = []

start = datetime.now()
#print("start: ", start)

sim = glob(output_path+'systems-recovery/transits0_0_0_0.csv')
cube = prior_grid_logslope(cube, ndim, nparams, 0, 0, 0)
transit_multiplicities = []
geom_transit_multiplicities = []
intact_fracs = []
disrupted_fracs = []
logLs = []

output = pd.DataFrame()

# cycle through different fractions of systems with planets
for f in np.linspace(0.1, 1, 10):

	for i in range(len(sim)):

		ms.append(cube[0])
		bs.append(cube[1])
		cs.append(cube[2])
		fs.append(f)
					
		df = pd.read_csv(sim[i], sep=',', on_bad_lines='skip')

		# populate future columns for output DataFrame
		transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs = collect(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs)

for gi_m in range(3):

	for gi_b in range(2):
		
		gi_b = gi_b + 1

		for gi_c in range(11):
			print(gi_m, gi_b, gi_c) # so I know where I am

			try:
				sim = glob(output_path+'systems-recovery/transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'*')
			except:
				print("file not found: ", 'transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'*')
				continue # if no file found, skip to next iteration 

			cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)
			
			# cycle through different fractions of systems with planets
			for f in np.round(np.linspace(0.1, 1, 10), 1):
				
				for i in range(len(sim)):
					#print("hi: ", gi_m, gi_b, gi_c, cube[0], cube[1], cube[2])
					ms.append(cube[0])
					bs.append(cube[1])
					cs.append(cube[2])
					fs.append(f)

					df = pd.read_csv(sim[i], sep=',', on_bad_lines='skip')

					# populate future columns for output DataFrame
					transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs = collect(df, f, transit_multiplicities, geom_transit_multiplicities, intact_fracs, disrupted_fracs, logLs)

#end = datetime.now()
#print("TIME ELAPSED: ", end-start)

df_logL = pd.DataFrame({'ms': ms, 'bs': bs, 'cs': cs, 'fs': fs, 
	'transit_multiplicities': transit_multiplicities, 'geom_transit_multiplicities': geom_transit_multiplicities, 'logLs': logLs, 
	'intact_fracs': intact_fracs, 'disrupted_fracs': disrupted_fracs})
print(df_logL)

df_logL.to_csv(output_path+'collect_recovery.csv', index=False)

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
