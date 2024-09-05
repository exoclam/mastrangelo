##################################################
### Generate science plots for Paper III #########
##################################################

import os
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
from numpyro_ext.distributions import MixtureGeneral
from tqdm import tqdm
from ast import literal_eval
import seaborn as sns

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

def literal_eval_w_exceptions(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        pass

#path = '/blue/sarahballard/c.lam/sculpting2/'
path = '/Users/chrislam/Desktop/mastrangelo/' # new computer has different username

berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
#berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk_zink.csv') # stellar sample as (nearly) prescribed in Zink+ 2023

# make berger_kepler more wieldy
berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
						     'iso_age', 'iso_age_err1', 'iso_age_err2', 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'rrmscdpp06p0', 'iso_rad', 'iso_rad_err1', 'iso_rad_err2']]
#pnum = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv') # planet hosts among crossmatched Berger sample

# mise en place
k = pd.Series([833, 134, 38, 15, 5, 0])
k_score = pd.Series([631, 115, 32, 10, 4, 0])
k_fpp = pd.Series([1088, 115, 34, 9, 3, 0])
G = 6.6743e-8 # gravitational constant in cgs

period_grid = np.logspace(np.log10(2), np.log10(300), 10)
radius_grid = np.linspace(1, 4, 10)
height_bins = np.array([0., 150, 250, 400, 650, 3000])

# create JAX random seed
key = jax.random.key(42)

### Create synthetic Population

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

#print(simulate_helpers.draw_planet_radii(stats.loguniform.rvs(2, 300, size=1000)))

### model hyperparameters
"""
# List o' models, back when we were trying individual models
# 2 Gyr; 0.4; 0.05 --> f = 0.1
# 3 Gyr; 0.5; 0.2 --> f = 0.28
# 3 Gyr; 0.6; 0.25 --> f = 0.34
threshold = 3. 
frac1 = 0.6
frac2 = 0.25

# does 0.4 < f < 0.5? I'll allow down to 0.3 as well (Zhu+ 2018)
#pop1 = len(berger_kepler.loc[berger_kepler['iso_age'] < threshold]) * frac1
#pop2 = len(berger_kepler.loc[berger_kepler['iso_age'] >= threshold]) * frac2
#print("f: ", (pop1+pop2)/len(berger_kepler))
#quit()
"""

#valid_models = simulate_helpers.generate_models(berger_kepler)
#valid_models.to_csv(path+'galactic-occurrence/data/valid_models.csv', index=False)
valid_models = pd.read_csv(path+'galactic-occurrence/data/valid_models.csv')
### visualize models
#simulate_helpers.plot_galactic_occurrence_models(valid_models)

### choose representative models to try
model_sample = simulate_helpers.subsample_models(valid_models)

x = np.linspace(0, 12, 100)
for model_key in range(len(model_sample)):
    model = model_sample.iloc[model_key]
    threshold = model['threshold']
    frac1 = model['f1']
    frac2 = model['f2']

    #os.mkdir(path+'galactic-occurrence/data/') 

    physical_planet_occurrences = []
    detected_planet_occurrences_all = []
    adjusted_planet_occurrences_all = []
    transit_multiplicities_all = []
    geom_transit_multiplicities_all = []
    completeness_all = []
    # for each model, draw around stellar age errors 10 times
    for j in range(10):

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

        # create a Population object to hold information about the occurrence law governing that specific population
        pop = Population(berger_kepler_temp['kepid'], berger_kepler_temp['age'], threshold, frac1, frac2)
        frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)

        # create Star objects, with their planetary systems
        star_data = []
        for i in tqdm(range(len(berger_kepler))): # 100

            #new_key, subkey = jax.random.split(key)
            #del key  # The old key is consumed by split() -- we must never use it again.

            #val = jax.random.normal(subkey)
            #del subkey  # The subkey is consumed by normal().

            #key = new_key  # new_key is safe to use in the next iteration.

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

        ### Calculate occurrence rates and compare over galactic heights, a la Zink+ 2023 Fig 12
        zink_k2 = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500.]), 'occurrence': np.array([45, 37, 34, 12]), 'occurrence_err1': np.array([21, 12, 11, 5]), 'occurrence_err2': np.array([15, 11, 8, 5])})
        zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})

        # bin systems by galactic height
        berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
        berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['kepid'])

        # turn off usually; just for testing
        #berger_kepler_stars1 = berger_kepler_all.loc[berger_kepler_all['height'] <= 150]
        #berger_kepler_stars2 = berger_kepler_all.loc[(berger_kepler_all['height'] > 150) & (berger_kepler_all['height'] <= 250)]
        #berger_kepler_stars3 = berger_kepler_all.loc[(berger_kepler_all['height'] > 250) & (berger_kepler_all['height'] <= 400)]
        #berger_kepler_stars4 = berger_kepler_all.loc[(berger_kepler_all['height'] > 400) & (berger_kepler_all['height'] <= 650)]
        #berger_kepler_stars5 = berger_kepler_all.loc[berger_kepler_all['height'] > 650]

        # isolate planet hosts and bin them by galactic height
        berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
        berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas'])
        berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
        berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 2.] # limit radii to fairly compare with SEs in Zink+ 2023
        berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

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
        physical_planet_occurrence = berger_kepler_planets_counts/berger_kepler_counts
        #print(physical_planet_occurrence)
        #print("physical planet occurrence rates, per 100 stars: ", 100*physical_planet_occurrence)
        physical_planet_occurrences.append(100*physical_planet_occurrence)

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

    #detected_planet_occurrences_all = 100 * np.array(detected_planet_occurrences_all)
    mean_transit_multiplicities = 100*np.mean(transit_multiplicities_all, axis=0)
    print("mean transit multiplicities across all systems: ", mean_transit_multiplicities)
    mean_geom_transit_multiplicities = 100*np.mean(geom_transit_multiplicities_all, axis=0)
    mean_detected_planet_occurrences = 100*np.mean(detected_planet_occurrences_all, axis=0)
    print("scale heights: ", zink_kepler['scale_height'])
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
    quit()

    ### plot physical occurrences vs galactic height, to compare against Zink+ 2023
    f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
    z_max = np.logspace(2, 3, 100)
    metallicity_trend = 100 * 0.63 * (10**(-0.14*np.linspace(-0.5, 0.5, 100))) * 0.5

    plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data', alpha=0.5)
    plt.errorbar(x=zink_kepler['scale_height'], y=np.mean(physical_planet_occurrences, axis=0), yerr=np.std(physical_planet_occurrences, axis=0), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='red', alpha=0.5, label='model [actual]')
    plt.plot(z_max, metallicity_trend, label='metallicity trend', color='green', alpha=0.4, linestyle='--')

    plt.xlim([100, 1000])
    plt.ylim([6, 100])
    plt.xscale('log')
    plt.yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.xticks(ticks=[100,300, 1000])
    plt.yticks(ticks=[10, 30, 100])
    plt.xlabel("galactic scale height [pc]")
    plt.ylabel("planets per 100 stars")
    plt.title('f=%1.1f' % frac1 + ' if <=%i ' % threshold + 'Gyr; f=%1.2f' % frac2 + ' if >%i ' % threshold + 'Gyr') 
    plt.legend(loc='upper left', bbox_to_anchor=[1.0, 1.05])
    plt.tight_layout()

    # reformat plot title name
    thresh_str = '_%1.1f' % threshold
    thresh_str = thresh_str.replace('.','p')
    frac1_str = '_%1.1f' % frac1
    frac1_str = frac1_str.replace('.','p')
    frac2_str = '_%1.1f' % frac2
    frac2_str = frac2_str.replace('.','p')

    plt.savefig(path+'galactic-occurrence/plots/model_vs_zink' + thresh_str + frac1_str + frac2_str + '.png', format='png', bbox_inches='tight')


"""
Cell by cell completeness over radius and period space
"""
"""
# several cells have uncertainties of 0% because there is only one surviving non-NaN realization; get rid of those, too 
# some of them still round to 0%, though
mean_completeness_map[std_completeness_map == 0] = np.nan
std_completeness_map[std_completeness_map == 0] = np.nan

# plot
f, ((ax1)) = plt.subplots(1, 1, figsize=(8, 8))
formatted_text = (np.asarray(["{0}±{1}%".format( 
    np.round(100*mean, 1), np.round(100*std, 1)) for mean, std in zip(mean_completeness_map.flatten(), std_completeness_map.flatten())])).reshape(9, 9) 
sns.heatmap(mean_completeness_map, yticklabels=np.around(radius_grid, 1), xticklabels=np.around(period_grid, 0), vmin=0., vmax=1., cmap='Blues', cbar_kws={'label': 'completeness'}, annot=formatted_text, fmt="", annot_kws={"size": 7})
ax1.set_xticks(ax1.get_xticks()[::2]) # sample every other tick, for cleanness
ax1.set_yticks(ax1.get_yticks()[::2]) # sample every other tick, for cleanness
ax1.invert_yaxis()
plt.xlabel('period [days]')
plt.ylabel('radius [$R_{\oplus}$]')
#plt.xticks(ticks=period_grid)
#plt.yticks(ticks=radius_grid)
f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(path+'galactic-occurrence/plots/completeness.png')
"""

"""
Compare model age vs multiplicity yield against Yang+ 2023 result from PAST IV paper
https://iopscience.iop.org/article/10.3847/1538-3881/ad0368#ajad0368f4
"""
"""
yang_comparison = pd.DataFrame({'multi': ['0', '1', '2', '3+'], 
                               'age': [4.09, 3.27, 2.77, 2.20],
                               'err1': [0.44, 0.30, 0.22, 0.16],
                               'err2': [0.35, 0.26, 0.22, 0.16]})

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as colors

blue_patch = mpatches.Patch(color='#fedbcc', label='model yield')

total, transit, nontransit = simulate_helpers.collect_age_histograms(berger_kepler_planets) # berger_kepler_planets or berger_kepler_transiters?
nontransit = nontransit.drop_duplicates(subset='kepid')
transit = transit.drop_duplicates(subset='kepid')
total = pd.concat([transit, nontransit])
total['group'] = 1
total.loc[total["yield"] > 3, "yield"] = 3
#print(np.unique(total['yield']))
#print(total[total['yield']>3]['age'])
#print("0: ", len(total.loc[total['yield']==0]))
#print("1: ", len(total.loc[total['yield']==1]))
#print("2: ", len(total.loc[total['yield']==2]))
#print("3+: ", len(total.loc[total['yield']>=3]))

ax1 = sns.boxplot(data=pd.concat([total]), x='yield', y='age', hue='group', 
            orient="v", whis=(5,95), showfliers=False, palette=['#fedbcc', '#89bedc'], zorder=1)
ax3 = plt.errorbar([0,1,2,3], yang_comparison.age, yerr=[yang_comparison.err2, yang_comparison.err1],
    fmt='o', capsize=5, color='k', label='Yang+ 2023', zorder=2)

for patch in ax1.artists:
    fc = patch.get_facecolor()
    patch.set_facecolor(colors.to_rgba(fc, 0.6))
    
plt.ylim([0,14])
plt.xticks([0,1,2,3], ['0','1','2','3+'])
plt.xlim([-0.5, 3.5])
plt.xlabel('multiplicity')
plt.ylabel('age [Gyrs]')
plt.legend(handles=[blue_patch, ax3], loc='upper left', bbox_to_anchor=[1.0, 1.05])
plt.tight_layout()
plt.savefig(path+'galactic-occurrence/plots/age_vs_multiplicity_boxplot.png', format='png', bbox_inches='tight')
plt.show()
"""

"""
Fit a model through these points to get best-fit parameters for comparison with Zink+ 2023
"""

def linear_model(x, yerr, y=None):
    # These are the parameters that we're fitting and we're required to define explicit
    # priors using distributions from the numpyro.distributions module.
    theta = numpyro.sample("theta", dist.Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0, 1))

    # Transformed parameters to be tracked during sampling 
    m = numpyro.deterministic("m", jnp.tan(theta))
    b = numpyro.deterministic("b", b_perp / jnp.cos(theta))

    # Then we specify the sampling distribution for the data, or the likelihood function.
    # Here we're using a numpyro.plate to indicate that the data are independent.
    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(m * x + b, yerr), obs=y)

def power_model(x, yerr, y=None):

    teff_sun = 5772
    gamma = -0.14
    power = (teff_sun/1000) * gamma
    kappa = 1

    tau = numpyro.sample("tau", dist.Normal(-0.5, 0.5))

    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(100 * kappa * (10**power) * x**tau, yerr), obs=y)

sampler = infer.MCMC(
    infer.NUTS(power_model), # linear_model
    num_warmup=2000,
    num_samples=8000,
    num_chains=2,
    progress_bar=True,
)

sampler.run(jax.random.PRNGKey(0), np.array(zink_kepler['scale_height']), yerr, y=mean_physical_planet_occurrences)
inf_data = az.from_numpyro(sampler)
print(az.summary(inf_data))
print("chain: ", inf_data.posterior.data_vars['tau'])

tau = inf_data.posterior.data_vars['tau'].mean().values
print("tau: ", tau)
tau_std = inf_data.posterior.data_vars['tau'].std().values
print("tau std: ", tau_std)

from scipy.optimize import curve_fit

popt, pcov = curve_fit(simulate_helpers.zink_model_simple, zink_kepler['scale_height'], mean_physical_planet_occurrences, p0=([-0.3]))
print("data: ", zink_kepler['scale_height'], zink_kepler['occurrence'])
print("curve fit: ", popt)

"""
m = inf_data.posterior.data_vars['m'].mean().values
b = inf_data.posterior.data_vars['b'].mean().values
print("slope: ", m)
print("intercept: ", b)
# I know it's a power law and that Fig 12 in Zink+ 2023 is a log-log plot, but comparing apples to apples, the y-intercept is 34.6 and the slope is -0.02. 
# Wait, it's actually 34.6 at 100 pc, so the actual y-intercept is 36.6

m_std = inf_data.posterior.data_vars['m'].std().values
b_std = inf_data.posterior.data_vars['b'].std().values
print("slope spread: ", m_std)
print("intercept spread: ", b_std)
"""

"""
Plot occurrence vs galactic height, as well as transit multiplicity detected yield
"""
f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))

# Zink+ 2023 model (Eqn 4 in their paper, but assuming independence wrt metallicity)
kappa = 1. # from Appendix B in Zink+ 2023
z_max = np.logspace(2, 3, 100)
zink_model, zink_model_upper, zink_model_lower = simulate_helpers.zink_model(kappa, z_max, radius_type='se')
zink_model_eyeball = 36.6 - 0.02 * np.linspace(100, 1000, 100)
metallicity_trend = 100 * 0.63 * (10**(-0.14*np.linspace(-0.5, 0.5, 100))) * 0.5

# my model
my_model = simulate_helpers.zink_model_simple(z_max, tau)
print("tau: ", tau)
print("my model: ", my_model)
quit()
my_model_upper = simulate_helpers.zink_model_simple(z_max, tau + tau_std)
my_model_lower = simulate_helpers.zink_model_simple(z_max, tau - tau_std)

my_model_curve_fit = simulate_helpers.zink_model_simple(z_max, popt)

#plt.errorbar(x=zink_k2['scale_height'], y=zink_k2['occurrence'], yerr=(zink_k2['occurrence_err1'], zink_k2['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='K2', alpha=0.5)
plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data', alpha=0.5)
#plt.scatter(x=zink_kepler['scale_height'], y=physical_planet_occurrence, color='red', label='model [actual]')
plt.errorbar(x=zink_kepler['scale_height'], y=np.mean(physical_planet_occurrences, axis=0), yerr=np.std(physical_planet_occurrences, axis=0), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='red', alpha=0.5, label='model [actual]')
#plt.scatter(x=zink_kepler['scale_height'], y=mean_adjusted_planet_occurrences_all, label='model [adjusted]')
#plt.scatter(x=zink_kepler['scale_height'], y=mean_detected_planet_occurrences, label='model [detected]')
###plt.plot(z_max, zink_model, label='Zink+ 2023 model (Fig 12)', color='orange')
#plt.plot(np.linspace(100, 1000, 100), zink_model_eyeball, label='Zink+ 2023 model (by eye)', color='orange', linestyle='--')
plt.plot(z_max, zink_model * 10, label='Zink+ 2023 model, x10', color='orange', linestyle='dotted')
plt.fill_between(z_max, zink_model_upper * 10, zink_model_lower * 10, alpha=0.3, color='orange')
plt.plot(z_max, metallicity_trend, label='metallicity trend', color='green', alpha=0.4, linestyle='--')
# intercept spread was  0.52 for 3 x 3 run

plt.plot(z_max, my_model_curve_fit, label='best-fit w/curve_fit', color='red')
#plt.fill_between(x=z_max, y1= my_model_upper, y2=my_model_lower, color='red', label='best-fit to model', alpha=0.3)

plt.xlim([100, 1000])
plt.ylim([6, 100])
plt.xscale('log')
plt.yscale('log')
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.xticks(ticks=[100,300, 1000])
plt.yticks(ticks=[10, 30, 100])
plt.xlabel("galactic scale height [pc]")
plt.ylabel("planets per 100 stars")
plt.title('f=%1.2f' % frac1 + ' if <=%i ' % threshold + 'Gyr; f=%1.2f' % frac2 + ' if >%i ' % threshold + 'Gyr') 
plt.legend(loc='upper left', bbox_to_anchor=[1.0, 1.05])
plt.tight_layout()
plt.savefig(path+'galactic-occurrence/plots/model_vs_zink3.png', format='png', bbox_inches='tight')
plt.show()
quit()

# also plot transit multiplicity against Kepler yield, as a diagnostic
print(np.max(transit_multiplicities_all, axis=0))

#plt.fill_between(np.arange(7)[1:], np.max(transit_multiplicities, axis=0), np.min(transit_multiplicities, axis=0), color='#03acb1', alpha=0.3, label='model detected') 
#plt.fill_between(np.arange(7)[1:], np.max(geom_transit_multiplicities, axis=0), np.min(geom_transit_multiplicities, axis=0), color='green', alpha=0.3, label='model geometric transit') 
plt.scatter(np.arange(7)[1:], k, color='r', marker='*', s=20, label='Kepler yield (koi_disposition)')
plt.xlabel('number of planets')
plt.ylabel('number of stars')
plt.legend()
plt.savefig(path+'galactic-occurrence/plots/transit-multiplicity6.png', facecolor='white', bbox_inches='tight')
plt.show()