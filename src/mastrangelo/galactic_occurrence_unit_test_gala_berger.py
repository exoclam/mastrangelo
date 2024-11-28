##################################################
### Testing specific models for Paper III ########
##################################################

import os
import os.path
import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform, gamma
from math import lgamma
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist, infer
import numpyro_ext
from numpyro_ext.distributions import MixtureGeneral
from tqdm import tqdm
from ast import literal_eval
import seaborn as sns

from itertools import zip_longest
import numpy.ma as ma # for masked arrays

from astropy.table import Table, join

from transit_class import Population, Star
import simulate_helpers
import simulate_transit
import collect_simulations 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import arviz as az
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)

"""
# troubleshoot JAX random key
key = jax.random.key(42)
for i in range(10):
    prob_intact = 0.18 + 0.1 * jax.random.truncated_normal(key=key, lower=0, upper=1)

    key, subkey = jax.random.split(key)
    print(prob_intact)
quit()
"""

planets = [ 785, 3156, 4996, 3196, 1357]
stars = [ 1885, 8321, 13565,  8510,  3636]
ratios = [0.41644562, 0.37928134, 0.36830077, 0.37555817, 0.37321232]
#print(np.sum(np.array(planets)))
#print(np.sum(np.array(stars)))
#print(np.sum(np.array(ratios)))

def literal_eval_w_exceptions(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        pass

path = '/Users/chrislam/Desktop/mastrangelo/' # new computer has different username

berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

"""
### TURN ON FOR BOUMA GYRO AGES 
bouma_ages = pd.read_csv(path+'data/table_star_gyro_agesformatted.csv') # Bouma+ 2024 gyrochronology ages
bouma_ages = bouma_ages.loc[bouma_ages['flag_gyro_quality'] > 10] # Bouma+ Sec 4.1 outlines quality flags for different science cases

# this is just for calculating scale height using gala, which requires the Bedell cross-match
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
megan = Table.read(path+'data/kepler_dr3_good.fits')
bouma = Table.read(path+'data/table_star_gyro_agesformatted.csv')

merged_berger_bedell = join(berger, megan, keys='kepid') # 70K stars
merged_berger_bouma = join(berger, bouma, keys='KIC') # 13.8K stars
merged_bouma_bedell = join(bouma, megan, keys_right='source_id', keys_left='dr3_source_id') # 20K stars

#merged = join(merged_bouma_bedell, megan, keys='kepid') # berger vs merged_berger_bouma vs merged_bedell_bouma
merged = join(merged_berger_bouma, megan, keys_right='source_id', keys_left='dr3_source_id')
merged.rename_column('kepid_2', 'kepid')
merged.rename_column('parallax_2', 'parallax')
merged.rename_column('feh_err1_2', 'feh_err1')
merged.rename_column('feh_err2_2', 'feh_err2')
#merged.rename_column('radius_2', 'radius')
#merged.rename_column('mass_2', 'mass')
#merged.rename_column('teff_2', 'teff')
#merged.rename_column('radius_err1_2', 'radius_err1')
#merged.rename_column('radius_err2_2', 'radius_err2')
#merged.rename_column('mass_err1_2', 'mass_err1')
#merged.rename_column('mass_err2_2', 'mass_err2')
#merged.rename_column('teff_err1_2', 'teff_err1')
#merged.rename_column('teff_err2_2', 'teff_err2')
berger_kepler = merged_berger_bedell.to_pandas() 
"""
berger_kepler = berger_kepler.drop_duplicates(subset=['kepid'])

# make berger_kepler more wieldy
#berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
#						     'iso_age', 'iso_age_err1', 'iso_age_err2', 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'rrmscdpp06p0', 'iso_rad', 'iso_rad_err1', 'iso_rad_err2']]

"""
### what happens to the Zink plot if it's just a narrow slice of cooler stars? 
merged_sub = merged_df.loc[(merged_df['iso_teff'] >= 5000) & (merged_df['iso_teff'] < 5500)]
"""

# enrich berger_kepler with z_maxes using gala. just needed to run that one time to output data/zmaxes.csv
#z_maxes = simulate_helpers.gala_galactic_heights(berger_kepler)
z_maxes = pd.read_csv(path+'data/zmaxes.csv')
berger_kepler['height'] = z_maxes * 1000

"""
### AGE AND ZMAX CHECKS
### Compare gala heights to Berger+ 2020 ages
berger_kepler = berger_kepler.dropna(subset=['height', 'iso_age', 'iso_age_err1', 'iso_age_err2']).reset_index() # 36K stars from 70K
heights = []
ages = [] 
for i in range(10):
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
    berger_kepler_temp = simulate_helpers.draw_galactic_heights(berger_kepler_temp)
    #heights.append(berger_kepler['height']) # for gala height
    heights.append(berger_kepler_temp['height']*1000) # for Ma+ 2017 height
    ages.append(berger_kepler_temp['age'])
heights_clean = np.concatenate(np.array(heights))
ages_clean = np.concatenate(np.array(ages))
temp_df = pd.DataFrame({'height': heights_clean, 'age': ages_clean})
temp_df = temp_df.loc[(temp_df['height'] <= 0.8) & (temp_df['age'] <= 5)] # 213K

plt.hist2d(temp_df['age'], temp_df['height'], bins=100)
plt.xlabel('Berger+ 2020 isochrone age [Gyr]')
plt.ylabel(r'gala-calculated $Z_{max}$ [kpc]')
#plt.xlim([0, 14])
#plt.ylim([0, 3])
plt.tight_layout()
#plt.savefig(path+'galactic-occurrence/plots/isochrone_age_vs_gala_heights_binned.png')
plt.show()
quit()

### Compare gala heights to Bouma+ 2024 ages
berger_kepler = berger_kepler.dropna(subset=['height', 'gyro_median', 'gyro_+1sigma', 'gyro_-1sigma']).reset_index() # we are dropping like 30K stars by doing this!
berger_kepler['gyro_median'] = berger_kepler['gyro_median']/1000
berger_kepler['gyro_+1sigma'] = berger_kepler['gyro_+1sigma']/1000
berger_kepler['gyro_-1sigma'] = berger_kepler['gyro_-1sigma']/1000
heights = []
ages = [] 
for i in range(10):
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'gyro_median', 'gyro_+1sigma', 'gyro_-1sigma', 'gyro_age')
    heights.append(berger_kepler['height'])
    ages.append(berger_kepler_temp['gyro_age'])
heights_clean = np.concatenate(np.array(heights))
ages_clean = np.concatenate(np.array(ages))

plt.hist2d(ages_clean, heights_clean, bins=100)
plt.xlabel('Bouma+ 2024 gyrochronological age [Gyr]')
plt.ylabel(r'gala-calculated $Z_{max}$ [kpc]')
plt.xlim([0, 4])
#plt.yscale('log')
plt.ylim([0, 2])
plt.tight_layout()
#plt.savefig(path+'galactic-occurrence/plots/gyro_age_vs_gala_heights_binned.png')
plt.show()
quit()
"""

"""
### TESTING BOUMA GYRO AGES VS GALACTIC HEIGHT
berger_kepler = berger_kepler.dropna(subset=['height', 'gyro_median', 'gyro_+1sigma', 'gyro_-1sigma']).reset_index() # we are dropping like 30K stars by doing this!
berger_kepler['gyro_median'] = berger_kepler['gyro_median']/1000
berger_kepler['gyro_+1sigma'] = berger_kepler['gyro_+1sigma']/1000
berger_kepler['gyro_-1sigma'] = berger_kepler['gyro_-1sigma']/1000
berger_kepler['height'] = berger_kepler['height']*1000
#print(list(berger_kepler.columns))
"""

# mise en place
k = pd.Series([833, 134, 38, 15, 5, 0])
k_score = pd.Series([631, 115, 32, 10, 4, 0])
k_fpp = pd.Series([1088, 115, 34, 9, 3, 0])
G = 6.6743e-8 # gravitational constant in cgs

period_grid = np.logspace(np.log10(2), np.log10(300), 10)
radius_grid = np.linspace(1, 4, 10)
height_bins = np.array([0., 150, 250, 400, 650, 3000])

# values from Zink+ 2023 Fig 12; errors combined by taking the sqrt(sum of squares)
zink_sn_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([38, 29, 23, 24, 17]), 'occurrence_err1': np.array([5, 3, 2, 2, 4]), 'occurrence_err2': np.array([6, 3, 2, 4, 4])})
zink_se_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})
zink_kepler_occurrence = np.array([38, 29, 23, 24, 17])+np.array([28, 29, 25, 27, 18])
zink_kepler_occurrence_err1 = np.round(np.sqrt((zink_sn_kepler['occurrence_err1'])**2 + (zink_se_kepler['occurrence_err1']**2)), 2)
zink_kepler_occurrence_err2 = np.round(np.sqrt((zink_sn_kepler['occurrence_err2'])**2 + (zink_se_kepler['occurrence_err2']**2)), 2)
zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': zink_kepler_occurrence, 'occurrence_err1': zink_kepler_occurrence_err1, 'occurrence_err2': zink_kepler_occurrence_err2})
#zink_kepler = zink_se_kepler # decide whether I want SEs only, SNs only, or both

# create JAX random seed
key = jax.random.key(42)

### Create synthetic Population

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

#print(simulate_helpers.draw_planet_radii(stats.loguniform.rvs(2, 300, size=1000)))

"""
Troubleshooting recreating Zink+ 2023 Fig 12 model, using model from Zink himself in email correspondence 20 Aug 2024
"""

"""
z_max = np.logspace(2, 3, 100)
def zink_model_simple(z_max, tau):

    teff_sun = 5772
    gamma = -0.14 # 0
    kappa = 1

    power = (teff_sun/1000) * gamma # feh and lam set to 0
    n = 100 * kappa * (10**power) * z_max**tau

    return n

zink_csv = pd.read_csv(path+'galactic-occurrence/data/SupEarths_combine_GaxScale_teff_fresh.csv')
zink_csv_sn = pd.read_csv(path+'galactic-occurrence/data/SubNeptunes_combine_GaxScale_teff_fresh.csv')
print("gamma stats: ", np.median(zink_csv['Gamma']), np.std(zink_csv['Gamma']))
print("gamma stats, SNs: ", np.median(zink_csv_sn['Gamma']), np.std(zink_csv_sn['Gamma']))
print("tau stats: ", np.median(zink_csv['Tau']), np.std(zink_csv['Tau']))
print("tau stats, SNs: ", np.median(zink_csv_sn['Tau']), np.std(zink_csv_sn['Tau']))

def model(x, tau, occurrence):
    #zink_csv = zink_csv*1000
    #tau = zink_csv['Tau']

    dln = 0.0011
    scaleMax= 1000
    scaleMin = 100
    const = (scaleMax)**(tau+1)/(tau+1) - ((scaleMin)**(tau+1)/(tau+1))
    y = occurrence * x**(tau)/const/dln
    
    return (y*100)

f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
for row in range(len(zink_csv)):
    tau = zink_csv['Tau'].iloc[row]
    occurrence = zink_csv['Occurrence'].iloc[row]
    gamma = zink_csv['Gamma'].iloc[row]

    if row==0:
        plt.plot(z_max, model(z_max, tau, occurrence), color='blue', alpha=0.2, label='posterior models')
    else:
        plt.plot(z_max, model(z_max, tau, occurrence), color='blue', alpha=0.2)

zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})
#plt.plot(z_max, model(z_max, -0.3, 0.225), color='red', alpha=0.8, label='approximate best-fit from Fig 12')
plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')

argmedian = np.argsort(zink_csv['Tau'])[len(zink_csv)//2]
argmedian_sn = np.argsort(zink_csv_sn['Tau'])[len(zink_csv_sn)//2]
#plt.plot(z_max, model(z_max, np.median(zink_csv['Tau']), zink_csv.iloc[argmedian]['Occurrence']), color='orange', alpha=0.8, label='posterior median Tau')
#argmedian = np.argsort(zink_csv['Occurrence'])[len(zink_csv)//2]
#argmedian_sn = np.argsort(zink_csv_sn['Occurrence'])[len(zink_csv_sn)//2]
print(zink_csv.iloc[argmedian]['Occurrence'])
print(zink_csv_sn.iloc[argmedian_sn]['Occurrence'])

#plt.plot(z_max, model(z_max, zink_csv.iloc[argmedian]['Tau'], np.median(zink_csv['Occurrence'])), color='orange', alpha=0.8, label='posterior median Occurrence, SEs')
plt.plot(z_max, model(z_max, np.median(zink_csv['Tau']), np.median(zink_csv['Occurrence'])), color='orange', alpha=0.8, label='posterior median Occurrence, SEs')
print(zink_csv.iloc[argmedian]['Tau'])
plt.plot(z_max, model(z_max, zink_csv_sn.iloc[argmedian_sn]['Tau'], np.median(zink_csv_sn['Occurrence'])), color='purple', alpha=0.8, label='posterior median Occurrence, SNs')
print(zink_csv_sn.iloc[argmedian_sn]['Tau'])

#metallicity_trend = 100 * 0.63 * (10**(-0.14*np.linspace(-0.5, 0.5, 100))) * 0.5
#plt.plot(z_max, metallicity_trend, color='green', linestyle='--', alpha=0.8, label='[Fe/H]')

plt.xlim([100, 1000])
plt.ylim([6, 100])
plt.yscale('log')
plt.xscale('log')
plt.ylabel('planets per 100 stars')
plt.xlabel('scale height [pc]')
plt.legend(loc='upper left', bbox_to_anchor=[1.0, 1.05])
plt.tight_layout()
plt.show()
quit()
"""

"""
Try models more reflective of peaked SFR from Garrison-Kimmmel Fig 11
"""
"""
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt

# Set-up
n = 1000
#numpy.random.seed(0x5eed)

# Parameters of the mixture components
def transform_abscissae(a, b, loc, scale):
    a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
    return a_transformed, b_transformed

### bumpy 1
a1, b1 = transform_abscissae(0, 14, 5, 0.5)
a2, b2 = transform_abscissae(0, 14, 7, 0.5)
a3, b3 = transform_abscissae(0, 14, 9, 3.5)
trunc_norm_params = np.array([[a1, b1, 5, 0.5],
                        [a2, b2, 7, 0.5],
                        [a3, b3, 9, 3.5]])

# Weight of each component
weights = np.array([0.05, 0.1, 0.85])

### bumpy 2
a1, b1 = transform_abscissae(0, 14, 7, 0.5)
a2, b2 = transform_abscissae(0, 14, 10, 0.5)
a3, b3 = transform_abscissae(0, 14, 8, 3.5)
trunc_norm_params = np.array([[a1, b1, 7, 0.5],
                        [a2, b2, 10, 0.5],
                        [a3, b3, 8, 3.5]])

# Weight of each component
weights = np.array([0.05, 0.1, 0.85])

# indices from which to choose the component
mixture_idx = numpy.random.choice(len(weights), size=n, replace=True, p=weights)

# y is the mixture sample
y = numpy.fromiter((ss.truncnorm.rvs(*(trunc_norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

# PDF support
xs = np.linspace(0, 14, 1000)

# generate PDF
ys = np.zeros_like(xs)
for (a, b, l, s), w in zip(trunc_norm_params, weights):
    ys += ss.truncnorm.pdf(xs, a=a, b=b, loc=l, scale=s) * w * 5
"""

"""
# Part 2
plt.plot(xs, ys)
plt.hist(y, density=True, bins="fd")
plt.xlim([0, 14])
plt.xlabel("cosmic age [Gyr]")
plt.ylabel("normalized SFR")
plt.show()

#print(xs, ys) # so, do 13.7 - xs
#quit()
"""

#"""

"""
Back to regular programming
"""
### model hyperparameters
# List o' models, back when we were trying individual models
# 2 Gyr; 0.4; 0.05 --> f = 0.1
# 3 Gyr; 0.5; 0.2 --> f = 0.28
# 3 Gyr; 0.6; 0.25 --> f = 0.34
# 3 gyr; 0.8; 0.27 --> f = 0.41
# 2 Gyr; 0.9; 0.2 --> f = 0.30
# 4 Gyr; 0.5; 0.5 --> f = 0.50
# 4 Gyr; 0.9; 0.01 --> f = 0.35

# 1) 1 Gyr; 0.5; 0.30 --> f = 0.30
# 2) 2 Gyr; 1.0; 0.20 --> f = 0.32
# 3) 6 Gyr; 0.8; 0.05 --> f = 0.48 (now 8 Gyr)
# 4) 8 Gyr; 0.65; 0.05 --> f = 0.50 (now 6 Gyr)
# 5) 2 Gyr; 1.; 0.3 --> f = 0.40 (now 12 Gyr)
# 6) 4 Gyr; 1.; 0.15 --> f = 0.47 (now 10 Gyr)
# bumpy:  f = 0.42
# monotonic: f = 0.29
# monotonic: y1 = 0.05, y2 = 0.7, f = 0.43
threshold = 10. # 13.7 minus stellar age, then round
frac1 = 0.15 # frac1 < frac2 if comparing cosmic ages
frac2 = 0.8

# does 0.4 < f < 0.5 (Lam & Ballard 2024)? I'll allow down to 0.3 as well (Zhu+ 2018)
#pop1 = len(berger_kepler.loc[berger_kepler['iso_age'] < threshold]) * frac1
#pop2 = len(berger_kepler.loc[berger_kepler['iso_age'] >= threshold]) * frac2
#print("f: ", (pop1+pop2)/len(berger_kepler))

physical_planet_occurrences = []
physical_planet_occurrences_precut = []
detected_planet_occurrences_all = []
adjusted_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
completeness_all = []
# for each model, draw around stellar age errors 10 times
for j in range(3): # 10

    #new_key, subkey = jax.random.split(key)
    #del key  # The old key is consumed by split() -- we must never use it again.

    #val = jax.random.normal(subkey)
    #del subkey  # The subkey is consumed by normal().  

    #key = new_key  # new_key is safe to use in the next iteration.
    key, subkey = jax.random.split(key)

    # draw stellar radii using asymmetric errors from Berger+ 2020 sample
    """
    #heights = []
    #ages = [] 
    #for i in range(10):
    #    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
    #    #berger_kepler_temp = simulate_helpers.draw_galactic_heights(berger_kepler_temp)
    #    heights.append(berger_kepler_temp['height']/1000) # for Ma+ 2017 height
    #    ages.append(berger_kepler_temp['age'])
    #heights_clean = np.concatenate(np.array(heights))
    #ages_clean = np.concatenate(np.array(ages))
    #temp_df = pd.DataFrame({'height': heights_clean, 'age': ages_clean})
    temp_df = pd.DataFrame({'height': np.array(trilegal_kepler['height']/1000), 'age': 10**(np.array(trilegal_kepler['logAge'])) / 1e9})
    temp_df = temp_df.loc[(temp_df['height'] <= 0.8) & (temp_df['age'] <= 5)] # 213K
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    norm = 10
    ax1.hist2d(temp_df['age'], temp_df['height'], bins=10, cmap='Blues')
    ax1.set_xlabel('TRILEGAL age [Gyr]')
    ax1.set_ylabel('TRILEGAL height [kpc]')
    ax1.set_title('unnormalized, TRILEGAL')
    hist, xedges, yedges = np.histogram2d(temp_df['age'], temp_df['height'], bins=10)
    hist = hist.T
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
        hist *= norm / hist.sum(axis=0, keepdims=True)
        hist *= norm / hist.sum(axis=1, keepdims=True)
    ax2.pcolormesh(xedges, yedges, hist, cmap='Blues')
    ax2.set_title('normalized, TRILEGAL')
    ax2.set_xlabel('TRILEGAL age [Gyr]')
    plt.tight_layout()
    plt.savefig(path+'galactic-occurrence/plots/trilegal_height_age.png') # ma17, gala, trilegal
    plt.show()
    quit()

    plt.hist2d(temp_df['age'], temp_df['height'], bins=10)
    plt.xlabel('Berger+ 2020 isochrone age [Gyr]')
    plt.ylabel('Ma+ 2017 age-dependent height [kpc]')
    #plt.xlim([0, 14])
    #plt.ylim([0, 3])
    plt.tight_layout()
    plt.savefig(path+'galactic-occurrence/plots/ma17_height_age.png')
    plt.show()
    quit()
    """


    heights = []
    ages = []
    for i in range(3):
        berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
        #berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'radius', 'radius_err1', 'radius_err2', 'stellar_radius')

        # draw stellar ages in the same way
        berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
        #berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'gyro_median', 'gyro_+1sigma', 'gyro_-1sigma', 'age')
        #berger_kepler_temp = simulate_helpers.draw_star_ages(berger_kepler) # in the future, I should read in from a fixed set of 30 pre-made DataFrames
        # but wait until we decide for sure whether to use Berger ages or some other age prescription

        # draw stellar masses in the same way
        berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')
        #berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'mass', 'mass_err1', 'mass_err2', 'stellar_mass')

        # draw galactic height based on age, using Ma+ 2017 relation
        #berger_kepler_temp = simulate_helpers.draw_galactic_heights(berger_kepler_temp)


        heights.append(berger_kepler_temp['height'])
        ages.append(berger_kepler_temp['age'])

    heights_clean = np.concatenate(np.array(heights))
    ages_clean = np.concatenate(np.array(ages))
    temp_df = pd.DataFrame({'height': heights_clean, 'age': ages_clean})
    temp_df = temp_df.loc[(temp_df['height'] <= 1500) & (temp_df['age'] <= 10)] 


    hist, xedges, yedges = np.histogram2d(temp_df['age'], temp_df['height'], bins=10)

    fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 4))
    hist = hist.T
    norm = 10
    #with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
        #hist *= norm / hist.sum(axis=0, keepdims=True)
        #hist *= norm / hist.sum(axis=1, keepdims=True)
    ax1.pcolormesh(xedges, yedges, hist, cmap='Blues')

    #plt.hist2d(berger_kepler_planets['age'], berger_kepler_planets['height'], bins=10, cmap='Blues')
    plt.ylabel('height [pc]')
    plt.xlabel('age [Gyr]')
    plt.tight_layout()
    plt.show()        
    quit()



    ### create a Population object to hold information about the occurrence law governing that specific population
    # STEP
    #pop = Population(10**trilegal_kepler['logAge']/1e9, threshold, frac1, frac2)
    pop = Population(berger_kepler_temp['age'], threshold, frac1, frac2)
    frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)

    # BUMPY
    #pop = Population(berger_kepler_temp['kepid'], berger_kepler_temp['age'])
    #frac_hosts = pop.galactic_occurrence_bumpy(xs, ys)

    # MONOTONIC
    #pop = Population(berger_kepler_temp['kepid'], berger_kepler_temp['age'])
    #y1 = 0.05 # 0.15
    #y2 = 0.8 # 0.75
    #frac_hosts = pop.galactic_occurrence_monotonic(y1, y2)

    # PIECEWISE
    #y1 = 0.05 
    #y2 = 0.95
    #threshold = 3.
    #pop = Population(berger_kepler_temp['kepid'], berger_kepler_temp['age'])
    #frac_hosts = pop.galactic_occurrence_piecewise(y1, y2, threshold)
    print("MEAN FRAC HOSTS: ", np.mean(frac_hosts))

    alpha_se = np.random.normal(-1., 0.2)
    alpha_sn = np.random.normal(-1.5, 0.1)

    # create Star objects, with their planetary systems
    star_data = []
    #for i in tqdm(range(len(trilegal_kepler))): # 100
    #for i in tqdm(range(10000)):
    for i in tqdm(range(len(berger_kepler))): # 100
        star = Star(berger_kepler_temp['age'][i], berger_kepler_temp['stellar_radius'][i], berger_kepler_temp['stellar_mass'][i], berger_kepler_temp['rrmscdpp06p0'][i], frac_hosts[i], berger_kepler_temp['height'][i], subkey, alpha_se, alpha_sn, berger_kepler_temp['kepid'][i])
        #star = Star(10**trilegal_kepler['logAge'][i], trilegal_kepler['stellar_radius'][i], trilegal_kepler['Mact'][i], trilegal_kepler['cdpp'][i], frac_hosts[i], trilegal_kepler['height'][i], subkey, alpha_se, alpha_sn, kepid=int(i))
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
            'omegas': star.omegas,
            'planet_radii': star.planet_radii
        }
        star_data.append(star_update)
        pop.add_child(star)

    # output simulated planetary systems
    #print(star_data)
    #print(pop)

    ### Convert to Pandas
    berger_kepler_all = pd.DataFrame.from_records(star_data)
    berger_kepler_all['height'] = berger_kepler_all['height'].astype(float)
    print("BEFORE DROPPING HEIGHTLESS STARS: ", 100*np.nansum(berger_kepler_all.num_planets)/len(berger_kepler_all))
    berger_kepler_all = berger_kepler_all.dropna(subset='height').reset_index()
    print("AFTER DROPPING HEIGHTLESS STARS: ", 100*np.nansum(berger_kepler_all.num_planets)/len(berger_kepler_all))

    #berger_kepler_all = berger_kepler_all.replace('  ', ',').replace('\r\n ', ',')
    #berger_kepler_all = ast.literal_eval(dat2)
    ###berger_kepler_all.to_csv(path+'galactic-occurrence/systems/3gyr_p5_p2/berger_kepler_planets_'+str(j)+'.csv', index=False)

    """
    Assign galactic heights, transit status, and detected planets for each system.
    """
    # read in non-exploded generated system data, which includes non-planet hosts
    ###berger_kepler_all = pd.read_csv(path+'galactic-occurrence/systems/3gyr_p6_p25/berger_kepler_planets_'+str(i)+'.csv')
    berger_kepler_all['periods'] = berger_kepler_all['periods'].apply(literal_eval_w_exceptions)
    berger_kepler_all['planet_radii'] = berger_kepler_all['planet_radii'].apply(literal_eval_w_exceptions)
    berger_kepler_all['incls'] = berger_kepler_all['incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['mutual_incls'] = berger_kepler_all['mutual_incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['eccs'] = berger_kepler_all['eccs'].apply(literal_eval_w_exceptions)
    berger_kepler_all['omegas'] = berger_kepler_all['omegas'].apply(literal_eval_w_exceptions)
    print(len(berger_kepler), len(berger_kepler_all))
    #print(100*np.nansum(berger_kepler_all.num_planets)/len(berger_kepler_all))

    """
    # make Figure 1 for paper: color-coded ages by scale height for a fiducial stellar sample
    f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
    plt.scatter(berger_kepler_all.reset_index().index, berger_kepler_all.height/1000, c=berger_kepler_all.age, s=5, alpha=0.8)

    plt.hlines(0.3, min(berger_kepler_all.reset_index().index), max(berger_kepler_all.reset_index().index), linestyle='dotted', color='r', label='thin disk', linewidth=3)
    plt.hlines(1.1, min(berger_kepler_all.reset_index().index), max(berger_kepler_all.reset_index().index), linestyle='dashed', color='r', label='thick disk', linewidth=3)

    plt.ylabel('galactic scale height [kpc]')
    plt.xlabel('arbitrary planet index')
    plt.colorbar(label='stellar age [Gyr]')

    plt.legend(loc='upper left', bbox_to_anchor=[1.2, 1.05])
    plt.savefig(path+'galactic-occurrence/plots/age_by_scale_height.png', format='png', bbox_inches='tight')
    plt.tight_layout()

    thin = berger_kepler_all.loc[berger_kepler_all['height'] < 1100]
    thick = berger_kepler_all.loc[berger_kepler_all['height'] >= 1100]
    print("average age of thin disk stars: ", np.mean(thin['age']), np.std(thin['age']))
    print("average age of thick disk stars: ", np.mean(thick['age']), np.std(thick['age']))
    quit()
    """

    ### Calculate occurrence rates and compare over galactic heights, a la Zink+ 2023 Fig 12
    #zink_k2 = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500.]), 'occurrence': np.array([45, 37, 34, 12]), 'occurrence_err1': np.array([21, 12, 11, 5]), 'occurrence_err2': np.array([15, 11, 8, 5])})
    #zink_sn_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([38, 29, 23, 24, 17]), 'occurrence_err1': np.array([5, 3, 2, 2, 4]), 'occurrence_err2': np.array([6, 3, 2, 4, 4])})
    #zink_se_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})

    # bin systems by galactic height
    #height_bins = np.linspace(0, 3000, 20) # DO I NEED FINER BINS LIKE THIS? 
    berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
    berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['kepid'])
    print("ONE: ", berger_kepler_counts, np.sum(berger_kepler_counts))

    # turn off usually; just for testing
    #berger_kepler_stars1 = berger_kepler_all.loc[berger_kepler_all['height'] <= 150]
    #berger_kepler_stars2 = berger_kepler_all.loc[(berger_kepler_all['height'] > 150) & (berger_kepler_all['height'] <= 250)]
    #berger_kepler_stars3 = berger_kepler_all.loc[(berger_kepler_all['height'] > 250) & (berger_kepler_all['height'] <= 400)]
    #berger_kepler_stars4 = berger_kepler_all.loc[(berger_kepler_all['height'] > 400) & (berger_kepler_all['height'] <= 650)]
    #berger_kepler_stars5 = berger_kepler_all.loc[berger_kepler_all['height'] > 650]

    # isolate planet hosts and bin them by galactic height
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas']).reset_index(drop=True)
    #print("PLANETS BEFORE CUTS: ", np.nansum(berger_kepler_planets.drop_duplicates(subset='kepid').num_planets))
    berger_kepler_planets_counts_precut = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

    berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?
    #print("NUMBER OF PLANETS: ", len(berger_kepler_planets))
    #print("PLANETS AFTER CUTS: ", np.nansum(berger_kepler_planets.drop_duplicates(subset='kepid').num_planets))
    berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])
    #print("TWO: ", berger_kepler_planets_counts)
    #print("TWO: ", np.sum(berger_kepler_planets_counts)) # why is this number different from PLANETS AFTER CUTS?
    #print("THREE: ", np.sum(np.array(berger_kepler_planets.drop_duplicates(subset='kepid').groupby(['height_bins']).count().reset_index()['kepid'])))

    #print("TEST: PLANETS AFTER CUTS: ", berger_kepler_planets.drop_duplicates(subset='kepid').num_planets)
    #print("TEST: TWO: ", np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()))

    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['height'] < 1500]
    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['age'] < 10]
    hist, xedges, yedges = np.histogram2d(berger_kepler_planets['age'], berger_kepler_planets['height'], bins=10)

    fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 4))
    hist = hist.T
    norm = 10
    #with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
        #hist *= norm / hist.sum(axis=0, keepdims=True)
        #hist *= norm / hist.sum(axis=1, keepdims=True)
    ax1.pcolormesh(xedges, yedges, hist, cmap='Blues')

    #plt.hist2d(berger_kepler_planets['age'], berger_kepler_planets['height'], bins=10, cmap='Blues')
    plt.ylabel('height [pc]')
    plt.xlabel('age [Gyr]')
    plt.show()        
    quit()

    """
    print("RATIOS BINNED INTO HEIGHTS: ", berger_kepler_planets_counts/berger_kepler_counts)
    plt.scatter(height_bins[:-1], 100*berger_kepler_planets_counts/berger_kepler_counts)
    plt.show()
    
    ### WHY DO I GET A FLAT RELATION BETWEEN PLANET OCCURRENCE AND GALACTIC HEIGHT USING THE GALA-CALCULATED HEIGHTS
    ### ...especially when there is a soft relation between age and height
    age_bins = np.array([0, 2, 4, 6, 8, 10, 12])
    # denominator
    berger_kepler_all['age_bins'] = pd.cut(berger_kepler_all['age'], bins=age_bins, include_lowest=True)
    berger_kepler_counts = np.array(berger_kepler_all.groupby(['age_bins']).count().reset_index()['kepid'])
    # numerator
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas'])
    berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 2.] # limit radii to fairly compare with SEs in Zink+ 2023
    berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['age_bins']).count().reset_index()['kepid'])
    print("RATIOS BINNED INTO AGES: ", berger_kepler_planets_counts/berger_kepler_counts)
    plt.scatter(age_bins[:-1], berger_kepler_planets_counts/berger_kepler_counts)
    plt.show()
    quit()
    """

    # turn off usually; just for testing
    #berger_kepler_planets1 = berger_kepler_planets.loc[berger_kepler_planets['height'] <= 150]
    #berger_kepler_planets2 = berger_kepler_planets.loc[(berger_kepler_planets['height'] > 150) & (berger_kepler_planets['height'] <= 250)]
    #berger_kepler_planets3 = berger_kepler_planets.loc[(berger_kepler_planets['height'] > 250) & (berger_kepler_planets['height'] <= 400)]
    #berger_kepler_planets4 = berger_kepler_planets.loc[(berger_kepler_planets['height'] > 400) & (berger_kepler_planets['height'] <= 650)]
    #berger_kepler_planets5 = berger_kepler_planets.loc[berger_kepler_planets['height'] > 650]
    #print(len(berger_kepler_planets1), len(np.unique(berger_kepler_planets1['kepid'])))
    #print(berger_kepler_planets_counts)
    #print(len(berger_kepler_planets1)/len(berger_kepler_stars1))

    # calculate "true" planet occurrence
    print(berger_kepler_planets_counts)
    print(berger_kepler_counts)
    physical_planet_occurrence = berger_kepler_planets_counts/berger_kepler_counts # normally yes
    print("physical planet occurrence: ", physical_planet_occurrence)

    #physical_planet_occurrence = np.sum(berger_kepler_planets_counts)/np.sum(berger_kepler_counts)# normally no
    physical_planet_occurrence_precut = np.sum(berger_kepler_planets_counts_precut)/np.sum(berger_kepler_counts)
    print("PLANETS PER 100 STARS: ", 100*np.sum(berger_kepler_planets_counts)/np.sum(berger_kepler_counts))

    #print(physical_planet_occurrence)
    #print("physical planet occurrence rates, per 100 stars: ", 100*physical_planet_occurrence)
    physical_planet_occurrences.append(100*physical_planet_occurrence)
    physical_planet_occurrences_precut.append(100*physical_planet_occurrence_precut)
    print("PLANETS PER 100 STARS: ", np.sum(np.array(physical_planet_occurrences)))

    detected_planet_occurrences = []
    adjusted_planet_occurrences = []
    transit_multiplicities = []
    geom_transit_multiplicities = []

    for i in range(1):  # 10

        #berger_kepler_planets_temp = berger_kepler_planets

        ### Simulate detections from these synthetic systems
        prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(berger_kepler_planets.periods, 
                                        berger_kepler_planets.stellar_radius, berger_kepler_planets.planet_radii,
                                        berger_kepler_planets.eccs, 
                                        berger_kepler_planets.incls, 
                                        berger_kepler_planets.omegas, berger_kepler_planets.stellar_mass,
                                        berger_kepler_planets.rrmscdpp06p0, angle_flag=True) 

        berger_kepler_planets['transit_status'] = transit_statuses[0]
        berger_kepler_planets['prob_detections'] = prob_detections[0]
        berger_kepler_planets['sn'] = sn
        berger_kepler_planets['geom_transit_status'] = geom_transit_statuses

        """
        # Try the ExoMULT way (https://github.com/jonzink/ExoMult/blob/master/ScalingK2VIII/ExoMult.py)        
        completeness = probability_detection(berger_kepler_planets_temp.periods, 
                                        berger_kepler_planets_temp.stellar_radius, 2.*np.ones(len(berger_kepler_planets_temp)), # eventually I will draw planet radii
                                        berger_kepler_planets_temp.eccs, 
                                        berger_kepler_planets_temp.mutual_incls, 
                                        berger_kepler_planets_temp.omegas, berger_kepler_planets_temp.stellar_mass,
                                        berger_kepler_planets_temp.rrmscdpp06p0, angle_flag=True)
        """

        # need kepid to be str or tuple, else unhashable type when groupby.count()
        berger_kepler_planets['kepid'] = berger_kepler_planets['kepid'].apply(str) 

        # isolate detected transiting planets
        berger_kepler_transiters = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
        #print("physical planets: ", len((berger_kepler_planets['kepid'])))
        #print("detected planets: ", len((berger_kepler_transiters['kepid'])))
        
        # read out detected yields
        #berger_kepler_transiters.to_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv', index=False)

        # Read in pre-generated population
        #transiters_berger_kepler = pd.read_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv')

        ### Completeness
        # Calculate completeness map
        completeness_map, piv_physical, piv_detected = simulate_helpers.completeness(berger_kepler_planets, berger_kepler_transiters)
        completeness_threshold = 0.01 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
        completeness_map = completeness_map.mask(completeness_map < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
        completeness_all.append(completeness_map)
        ### this is to find the threshold beyond which I can fully recover the physical yield using the detected yield and completeness map
        #print("physical: ", simulate_helpers.adjust_for_completeness(berger_kepler_planets, completeness_map, radius_grid, period_grid, flag='physical'))
        #print("detected, adjusted: ", simulate_helpers.adjust_for_completeness(berger_kepler_transiters, completeness_map, radius_grid, period_grid, flag='detected'))

        ### Calculate transit multiplicity and other Population-wide demographics
        #simulate_helpers.collect_galactic(berger_kepler_planets)

        # compute transit multiplicity
        transit_multiplicity = berger_kepler_transiters.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
        transit_multiplicity = transit_multiplicity.to_list()
        transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
        transit_multiplicities_all.append(transit_multiplicity)

        # also calculate the geometric transit multiplicity
        geom_transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['geom_transit_status']==1]
        geom_transit_multiplicity = geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid
        geom_transit_multiplicity = geom_transit_multiplicity.to_list()
        geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k
        geom_transit_multiplicities_all.append(geom_transit_multiplicity)

        """
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
        physical_planet_occurrence_rate = len(berger_kepler_planets)/len(np.unique(berger_kepler['kepid']))
        detected_planet_occurrence_rate = len(transiters_berger_kepler)/len(np.unique(berger_kepler['kepid']))

        print("transit multiplicity: ", transit_multiplicity)
        print("geometric transit multiplicity: ", geom_transit_multiplicity)
        print("logL : ", logL)
        print("intact fraction (out of planet hosts): ", intact_frac)
        print("disrupted fraction (out of planet hosts): ", disrupted_frac)
        print("planet host fraction: ", pop_frac_host)
        print("planet occurrence rate: ", planet_occurrence_rate)
        """

        # calculate detected occurrence rate 
        berger_kepler_transiters_counts = np.array(berger_kepler_transiters.groupby(['height_bins']).count().reset_index()['kepid'])
        detected_planet_occurrence = berger_kepler_transiters_counts/berger_kepler_counts
        detected_planet_occurrences_all.append(detected_planet_occurrence)

        # same, but adjust for period- and radius-dependent completeness 
        berger_kepler_transiters1 = berger_kepler_transiters.loc[berger_kepler_transiters['height'] <= 150]
        berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 150) & (berger_kepler_transiters['height'] <= 250)]
        berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 250) & (berger_kepler_transiters['height'] <= 400)]
        berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 400) & (berger_kepler_transiters['height'] <= 650)]
        berger_kepler_transiters5 = berger_kepler_transiters.loc[berger_kepler_transiters['height'] > 650]
        #print(len(berger_kepler_planets))
        #print(len(berger_kepler_transiters))
        #print(simulate_helpers.adjust_for_completeness(berger_kepler_transiters, completeness_map, radius_grid, period_grid))

        len_berger_kepler_transiters1, _ = simulate_helpers.adjust_for_completeness(berger_kepler_transiters1, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters2, _ = simulate_helpers.adjust_for_completeness(berger_kepler_transiters2, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters3, _ = simulate_helpers.adjust_for_completeness(berger_kepler_transiters3, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters4, _ = simulate_helpers.adjust_for_completeness(berger_kepler_transiters4, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters5, _ = simulate_helpers.adjust_for_completeness(berger_kepler_transiters5, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters = np.array([len_berger_kepler_transiters1, len_berger_kepler_transiters2, len_berger_kepler_transiters3, len_berger_kepler_transiters4, len_berger_kepler_transiters5])
        
        adjusted_planet_occurrence = len_berger_kepler_transiters/berger_kepler_counts
        adjusted_planet_occurrences_all.append(adjusted_planet_occurrence)

    #transit_multiplicities_all.append(transit_multiplicities)
    #geom_transit_multiplicities_all.append(geom_transit_multiplicities)
    #detected_planet_occurrences_all.append(detected_planet_occurrences)

"""
print("YIELDS: ", physical_planet_occurrences)
#print("YIELDS, ACROSS ALL RUNS: ", np.sum(physical_planet_occurrences, axis=1))
plt.hist(physical_planet_occurrences, label='R<2; 1<P<40')
plt.hist(physical_planet_occurrences_precut, label='all small planets, 2<P<300')
plt.xlabel('planets per 100 stars')
plt.ylabel('count')
plt.legend()
plt.savefig(path+'galactic-occurrence/plots/planets_per_100_stars_control.png')
plt.show()
quit()
"""

#detected_planet_occurrences_all = 100 * np.array(detected_planet_occurrences_all)
mean_transit_multiplicities = 100*np.mean(transit_multiplicities_all, axis=0)
print("mean transit multiplicities across all systems: ", mean_transit_multiplicities)
mean_geom_transit_multiplicities = 100*np.mean(geom_transit_multiplicities_all, axis=0)
mean_detected_planet_occurrences = 100*np.mean(detected_planet_occurrences_all, axis=0)
print("mean detected planet occurrence: ", mean_detected_planet_occurrences)
mean_physical_planet_occurrences = np.mean(physical_planet_occurrences, axis=0)
print("mean physical planet occurrence: ", mean_physical_planet_occurrences)
yerr = np.std(physical_planet_occurrences, axis=0)
print("std of physical planet occurrences per height bin: ", yerr)
mean_adjusted_planet_occurrences_all = 100*np.mean(adjusted_planet_occurrences_all, axis=0)
print("mean adjusted planet occurrences: ", mean_adjusted_planet_occurrences_all)
mean_completeness_map = np.nanmean(completeness_all, axis=0)
#print("mean completeness map: ", mean_completeness_map)
std_completeness_map = np.nanstd(completeness_all, axis=0)
#print("std completeness map: ", std_completeness_map)

### plot physical occurrences vs galactic height, to compare against Zink+ 2023
#f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
z_max = np.logspace(2, 3, 100)
metallicity_trend = 100 * 0.63 * (10**(-0.14*np.linspace(-0.5, 0.5, 100))) * 0.5

zink_csv = pd.read_csv(path+'galactic-occurrence/data/SupEarths_combine_GaxScale_teff_fresh.csv')
zink_csv_sn = pd.read_csv(path+'galactic-occurrence/data/SubNeptunes_combine_GaxScale_teff_fresh.csv')

#print("occurrence stats: ", np.median(zink_csv['Occurrence']), np.std(zink_csv['Occurrence']))
#print("gamma stats: ", np.median(zink_csv['Gamma']), np.std(zink_csv['Gamma']))
#print("tau stats: ", np.median(zink_csv['Tau']), np.std(zink_csv['Tau']))

#plot_zink_range = [np.mean(zink_csv['Occurrence']) - np.std(zink_csv['Occurrence']), np.mean(zink_csv['Occurrence']) + np.std(zink_csv['Occurrence'])]
#plot_zink_posteriors = zink_csv.loc[(zink_csv['Occurrence'] >= plot_zink_range[0]) & (zink_csv['Occurrence'] <= plot_zink_range[1])]

def model(x, tau, occurrence):

    dln = 0.0011
    scaleMax= 1000
    scaleMin = 100
    const = (scaleMax)**(tau+1)/(tau+1) - ((scaleMin)**(tau+1)/(tau+1))
    planet_yield = occurrence * x**(tau)/const/dln * 100
    
    return planet_yield

### but first, fit a power law 
def power_model(x, yerr, y=None):

    tau = numpyro.sample("tau", dist.Uniform(-1., 0.))
    occurrence = numpyro.sample("occurrence", dist.Uniform(0.01, 1.))

    dln = 0.0011
    scaleMax= 1000
    scaleMin = 100
    const = (scaleMax)**(tau+1)/(tau+1) - ((scaleMin)**(tau+1)/(tau+1))
    planet_yield = occurrence * x**(tau)/const/dln * 100
    #print("planet yield: ", planet_yield)
    #print("yerr: ", yerr)
    #print("y: ", y)
    #print("sample model: ", model(z_max, tau, occurrence))
    
    with numpyro.plate("data", len(x)):
        numpyro.sample("planet_yield", dist.Normal(planet_yield, yerr), obs=y)

# find MAP solution
init_params = {
    "tau": -0.35,
    "occurrence": 0.3,
}
#run_optim = numpyro_ext.optim.optimize(
#    power_model,
#    init_strategy=numpyro.infer.init_to_value(values=init_params),
#)
run_optim = numpyro_ext.optim.optimize(
        power_model, init_strategy=numpyro.infer.init_to_median()
    )
opt_params = run_optim(jax.random.PRNGKey(5), np.array(zink_kepler['scale_height']), yerr, y=mean_physical_planet_occurrences)
print("opt params: ", opt_params)

# sample posteriors for best-fit model to simulated data
sampler = infer.MCMC(
    infer.NUTS(power_model, dense_mass=True,
        regularize_mass_matrix=False,
        init_strategy=numpyro.infer.init_to_value(values=opt_params)), 
    num_warmup=2000,
    num_samples=3000,
    num_chains=4,
    progress_bar=True,
)

sampler.run(jax.random.PRNGKey(0), np.array(zink_kepler['scale_height']), yerr, y=mean_physical_planet_occurrences)
inf_data = az.from_numpyro(sampler)
print(az.summary(inf_data))

"""
_ = az.plot_trace(
    inf_data,
    var_names=["tau", "occurrence"],
    backend_kwargs={"constrained_layout": True},
)
"""

tau_ours = inf_data.posterior.data_vars['tau'].mean().values
print("tau: ", tau_ours)
tau_std = inf_data.posterior.data_vars['tau'].std().values
print("tau std: ", tau_std)

occurrence_ours = inf_data.posterior.data_vars['occurrence'].mean().values
print("occurrence: ", occurrence_ours)
occurrence_std = inf_data.posterior.data_vars['occurrence'].std().values
print("occurrence std: ", occurrence_std)

### set up plotting
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
left, bottom, width, height = [0.16, 0.3, 0.15, 0.15]
ax2 = fig.add_axes([left, bottom, width, height])

#ax1.plot(z_max, model(z_max, init_params['tau'], init_params['occurrence']), color='k', label='opt params model')

# metallicity trend
#ax1.plot(z_max, metallicity_trend, label='metallicity trend', color='green', alpha=0.4, linestyle='--')

# zink data
ax1.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='red', label='Zink+ 2023 Kepler data', alpha=0.5)

# zink model
# calculate all models so that we can take one-sigma envelope
yield_max = []
yield_min = []
models_se = []
models_sn = []
for i in range(len(zink_csv)):
    row = zink_csv.iloc[i]
    models_se.append(model(z_max, row['Tau'], row['Occurrence']))
zink_csv['model'] = models_se

for j in range(len(zink_csv_sn)):
    row = zink_csv_sn.iloc[i]
    models_sn.append(model(z_max, row['Tau'], row['Occurrence']))
zink_csv_sn['model'] = models_sn
sum_model = zink_csv['model'] + zink_csv_sn['model']
for temp_list in zip_longest(*sum_model):
    yield_max.append(np.percentile(temp_list, 84)) # plus one sigma
    yield_min.append(np.percentile(temp_list, 16)) # minus one sigma
ax1.fill_between(z_max, yield_max, yield_min, color='red', alpha=0.3, label='Zink+ 2023 posteriors') #03acb1

# our simulated data
ax1.errorbar(x=zink_kepler['scale_height'], y=np.mean(physical_planet_occurrences, axis=0), yerr=np.std(physical_planet_occurrences, axis=0), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='#03acb1', alpha=0.5, label='model yield')

# plot our best fit posteriors
our_yield_max = []
our_yield_min = []
our_models = []
for j in range(len(inf_data.posterior.data_vars['occurrence'])):

    tau = 0.5 * (inf_data.posterior.data_vars['tau'].values[0][j] + inf_data.posterior.data_vars['tau'].values[1][j])
    occurrence = 0.5 * (inf_data.posterior.data_vars['occurrence'].values[0][j] + inf_data.posterior.data_vars['occurrence'].values[1][j])
    #tau = inf_data.posterior.data_vars['tau'].values[0][j]
    #occurrence = inf_data.posterior.data_vars['occurrence'].values[0][j] 
    #print(z_max, tau, occurrence)
    #quit()
    our_models.append(model(z_max, tau, occurrence))
for temp_list2 in zip_longest(*our_models):
    our_yield_max.append(np.percentile(temp_list2, 84)) # plus one sigma
    our_yield_min.append(np.percentile(temp_list2, 16)) # minus one sigma
print("OUR YIELD: ", our_models)
print(len(our_models))
ax1.fill_between(z_max, our_yield_max, our_yield_min, color='#03acb1', alpha=0.3, label='model best-fit posteriors') 

#ax1.plot(z_max, model(z_max, tau_ours, occurrence_ours), color='#03acb1', label='mean model')
"""
# troubleshoot blue envelope
troubleshoot_models = []
troubleshoot_max = []
troubleshoot_min = []
for test_i in np.linspace(tau_ours - tau_std, tau_ours + tau_std, 10):
    for test_j in np.linspace(occurrence_ours - occurrence_std, occurrence_ours + occurrence_std, 10):
        troubleshoot_models.append(model(z_max, test_i, test_j))
for temp_list3 in zip_longest(*troubleshoot_models):
    troubleshoot_max.append(np.percentile(temp_list3, 84)) # plus one sigma
    troubleshoot_min.append(np.percentile(temp_list3, 16)) # minus one sigma     
ax1.fill_between(z_max, troubleshoot_max, troubleshoot_min, color='k', alpha=0.3, label='troubleshoot models') 
"""

ax1.set_xlim([100, 1000])
ax1.set_ylim([6, 100])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.set_xticks(ticks=[100, 300, 1000])
ax1.set_yticks(ticks=[10, 30, 100])
ax1.set_xlabel("galactic scale height [pc]")
ax1.set_ylabel("planets per 100 stars")
#plt.title('m12z-like SFH')
#ax1.set_title('f=%1.2f' % frac1 + ' if <=%i ' % threshold + 'Gyr; f=%1.2f' % frac2 + ' if >%i ' % threshold + 'Gyr') 
ax1.legend(loc='upper left', bbox_to_anchor=[1.0, 1.05])

# plot inset of model
"""
ax2.plot(x, y, color='powderblue')
ax2.set_xlabel('stellar age [Gyr]')
ax2.set_ylabel('host fraction')
ax2.set_ylim([0, 1.05])
"""

# step model
x = np.linspace(0, 14, 1000)
y = np.where(x <= threshold, frac1, frac2)

"""
# monotonic inset
x = np.linspace(0, 14, 1000)
m = (y2-y1)/14.
y = m * x + y1
"""

"""
# piecewise inset
x = np.linspace(0, 14, 1000)
m = (y2-y1)/(14 - threshold)
y = np.where(x < threshold, y1, y1 + m * (x-threshold))
"""
ax2.plot(x, y, color='powderblue')
ax2.set_xlabel('cosmic age [Gyr]')
ax2.set_ylabel('planet host fraction')
ax2.set_ylim([0,1])

fig.tight_layout()
plt.savefig(path+'galactic-occurrence/plots/model_vs_zink_step6_gala.png', format='png', bbox_inches='tight')
plt.show()

"""
# reformat plot title name
thresh_str = '_%1.1f' % threshold
thresh_str = thresh_str.replace('.','p')
frac1_str = '_%1.1f' % frac1
frac1_str = frac1_str.replace('.','p')
frac2_str = '_%1.1f' % frac2
frac2_str = frac2_str.replace('.','p')
plt.savefig(path+'galactic-occurrence/plots/model_vs_zink' + thresh_str + frac1_str + frac2_str + '.png', format='png', bbox_inches='tight')
"""