##################################################
### Helper functions, post-pop-synth #############
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
import jax
import jax.numpy as jnp
from tqdm import tqdm

from transit_class import Population, Star
import simulate_helpers
import simulate_transit
import collect_simulations 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)

#path = '/blue/sarahballard/c.lam/sculpting2/'
path = '/Users/chrislam/Desktop/mastrangelo/' # new computer has different username

berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# make berger_kepler more wieldy
berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
						     'iso_age', 'iso_age_err1', 'iso_age_err2', 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'rrmscdpp06p0', 'iso_rad', 'iso_rad_err1', 'iso_rad_err2']]
#pnum = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv') # planet hosts among crossmatched Berger sample
#k = pnum.groupby('kepid').count().koi_count.reset_index().groupby('koi_count').count()
k = pd.Series([833, 134, 38, 15, 5, 0])
k_score = pd.Series([631, 115, 32, 10, 4, 0])
k_fpp = pd.Series([1088, 115, 34, 9, 3, 0])
G = 6.6743e-8 # gravitational constant in cgs

# how many params, how many dims, initialize cube
ndim = 3
nparams = 3
cube = [0, 0, 0]

# create JAX random seed
key = jax.random.key(42)

### Create synthetic Population

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# draw stellar radii using asymmetric errors from Berger+ 2020 sample
berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')

# draw stellar ages in the same way
berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
#berger_kepler_temp = draw_star_ages(berger_kepler) # in the future, I should read in from a fixed set of 30 pre-made DataFrames
# but wait until we decide for sure whether to use Berger ages or some other age prescription

# draw stellar masses in the same way
berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')

# draw galactic height based on age, using Ma+ 2017 relation
berger_kepler_temp = simulate_helpers.draw_galactic_heights(berger_kepler_temp)

### model hyperparameters
threshold = 2. 
frac1 = 0.4
frac2 = 0.05

# create a Population object to hold information about the occurrence law governing that specific population
pop = Population(berger_kepler_temp['kepid'], berger_kepler_temp['age'], threshold, frac1, frac2)
frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)

# create Star objects, with their planetary systems
star_data = []
for i in tqdm(range(len(berger_kepler_temp))): # 100
    star = Star(berger_kepler_temp['kepid'][i], berger_kepler_temp['age'][i], berger_kepler_temp['stellar_radius'][i], berger_kepler_temp['stellar_mass'][i], berger_kepler_temp['rrmscdpp06p0'][i], frac_hosts[i], berger_kepler_temp['height'][i])
    star_update = {
        'kepid': star.kepid,
        'age': star.age,
        'stellar_radius': star.stellar_radius,
        'stellar_mass': star.stellar_mass,
        'rrmscdpp06p0': star.rrmscdpp06p0,
        'frac_host': star.frac_host,
        'height': star.height,
        'midplane': star.midplane,
        'prob_intact': star.prob_intact,
        'status': star.status,
        'sigma_incl': star.sigma_incl,
        'num_planets': star.num_planets,
        'periods': star.periods,
        'incls': star.incls,
        'mutual_incls': star.mutual_incls,
        'eccs': star.eccs,
        'omegas': star.omegas
    }
    star_data.append(star_update)
    pop.add_child(star)

# output simulated planetary systems
#print(star_data)
print(pop)

### Convert to Pandas
berger_kepler_planets = pd.DataFrame.from_records(star_data)

# AT THIS STAGE, PERMANENTLY SAVE OUT THE POPULATION
berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['num_planets'] > 0]
berger_kepler_planets = berger_kepler_planets.explode(['periods', 'incls', 'mutual_incls', 'eccs', 'omegas'])

### Simulate detections from these synthetic systems
prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(berger_kepler_planets.periods, 
                                berger_kepler_planets.stellar_radius, 2.*np.ones(len(berger_kepler_planets)), # eventually I will draw planet radii
                                berger_kepler_planets.eccs, 
                                berger_kepler_planets.mutual_incls, 
                                berger_kepler_planets.omegas, berger_kepler_planets.stellar_mass,
                                berger_kepler_planets.rrmscdpp06p0, angle_flag=True) # was np.ones(len(berger_kepler_planets))*131.4
        
berger_kepler_planets['transit_status'] = transit_statuses[0]
berger_kepler_planets['prob_detections'] = prob_detections[0]
berger_kepler_planets['sn'] = sn
berger_kepler_planets['geom_transit_status'] = geom_transit_statuses
print(berger_kepler_planets)
berger_kepler_planets.to_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected.csv', index=False)

# Read in pre-generated population
berger_kepler_planets = pd.read_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected.csv')

### Calculate transit multiplicity and other Population-wide demographics
#simulate_helpers.collect_galactic(berger_kepler_planets)

# isolate transiting planets
transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]

# compute transit multiplicity 
transit_multiplicity = transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
transit_multiplicity = transit_multiplicity.to_list()
transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k

# also calculate the geometric transit multiplicity
geom_transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['geom_transit_status']==1]
geom_transit_multiplicity = geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid
geom_transit_multiplicity = geom_transit_multiplicity.to_list()
geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k

# calculate logLs 
logL = collect_simulations.better_loglike(transit_multiplicity, k)
logL_score = collect_simulations.better_loglike(transit_multiplicity, k_score)
logL_fpp = collect_simulations.better_loglike(transit_multiplicity, k_fpp)

# get intact and disrupted fractions (among planet-hosts)
intact = berger_kepler_planets.loc[berger_kepler_planets['status']=='intact']
disrupted = berger_kepler_planets.loc[berger_kepler_planets['status']=='disrupted']
intact_frac = len(intact.kepid.unique())/len(berger_kepler_planets.kepid.unique())
disrupted_frac = len(disrupted.kepid.unique())/len(berger_kepler_planets.kepid.unique())
pop_frac_host = len(np.unique(berger_kepler_planets['kepid']))/len(np.unique(berger_kepler['kepid']))
planet_occurrence_rate = len(berger_kepler_planets)/len(np.unique(berger_kepler['kepid']))
print("transit multiplicity: ", transit_multiplicity)
print("geometric transit multiplicity: ", geom_transit_multiplicity)
print("logL : ", logL)
print("intact fraction (out of planet hosts: ", intact_frac)
print("disrupted fraction (out of planet hosts: ", disrupted_frac)
print("planet host fraction: ", pop_frac_host)
print("planet occurrence rate: ", planet_occurrence_rate)

### Calculate occurrence rates and compare over galactic heights, a la Zink+ 2023 Fig 12
zink_k2 = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500.]), 'occurrence': np.array([45, 37, 34, 12]), 'occurrence_err1': np.array([21, 12, 11, 5]), 'occurrence_err2': np.array([15, 11, 8, 5])})
zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})

# bin planet occurrence by galactic height
berger_kepler1 = berger_kepler_temp.loc[berger_kepler_temp['height'] < 150]
berger_kepler2 = berger_kepler_temp.loc[(berger_kepler_temp['height'] >= 150) & (berger_kepler_temp['height'] < 250)]
berger_kepler3 = berger_kepler_temp.loc[(berger_kepler_temp['height'] >= 250) & (berger_kepler_temp['height'] < 400)]
berger_kepler4 = berger_kepler_temp.loc[(berger_kepler_temp['height'] >= 400) & (berger_kepler_temp['height'] < 650)]
berger_kepler5 = berger_kepler_temp.loc[berger_kepler_temp['height'] >= 650]
print(len(berger_kepler1), len(berger_kepler2), len(berger_kepler3), len(berger_kepler4), len(berger_kepler5))

berger_kepler_planets1 = berger_kepler_planets.loc[berger_kepler_planets['height'] < 150]
berger_kepler_planets2 = berger_kepler_planets.loc[(berger_kepler_planets['height'] >= 150) & (berger_kepler_planets['height'] < 250)]
berger_kepler_planets3 = berger_kepler_planets.loc[(berger_kepler_planets['height'] >= 250) & (berger_kepler_planets['height'] < 400)]
berger_kepler_planets4 = berger_kepler_planets.loc[(berger_kepler_planets['height'] >= 400) & (berger_kepler_planets['height'] < 650)]
berger_kepler_planets5 = berger_kepler_planets.loc[berger_kepler_planets['height'] >= 650]
print(len(berger_kepler_planets1), len(berger_kepler_planets2), len(berger_kepler_planets3), len(berger_kepler_planets4), len(berger_kepler_planets5))

try:
    planet_occurrence1 = len(berger_kepler_planets1)/len(np.unique(berger_kepler1['kepid']))
except:
    planet_occurrence1 = 0
try:
    planet_occurrence2 = len(berger_kepler_planets2)/len(np.unique(berger_kepler2['kepid']))
except:
    planet_occurrence2 = 0
try:
    planet_occurrence3 = len(berger_kepler_planets3)/len(np.unique(berger_kepler3['kepid']))
except:
    planet_occurrence3 = 0
try:
    planet_occurrence4 = len(berger_kepler_planets4)/len(np.unique(berger_kepler4['kepid']))
except:
    planet_occurrence4 = 0
try:
    planet_occurrence5 = len(berger_kepler_planets5)/len(np.unique(berger_kepler5['kepid']))
except:
    planet_occurrence5 = 0
planet_occurrences = np.array([planet_occurrence1, planet_occurrence2, planet_occurrence3, planet_occurrence4, planet_occurrence5])
print("planet occurrences: ", planet_occurrences)

plt.errorbar(x=zink_k2['scale_height'], y=zink_k2['occurrence'], yerr=(zink_k2['occurrence_err1'], zink_k2['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='K2', alpha=0.5)
plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Kepler', alpha=0.5)
plt.scatter(x=zink_kepler['scale_height'], y=100*planet_occurrences, color='red', label='model')
plt.xlim([100, 1000])
plt.ylim([6, 100])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("galactic scale height [pc]")
plt.ylabel("planets per 100 stars")
plt.title('f=0.4 if <=2 Gyr; f=0.05 if >2 Gyr')
plt.legend()
plt.tight_layout()
plt.savefig(path+'galactic-occurrence/plots/test_model_vs_zink5.png')
plt.show()