##################################################
### Helper functions, post-pop-synth #############
##################################################

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

print(simulate_helpers.draw_planet_radii(stats.loguniform.rvs(2, 300, size=1000)))
quit()
"""
for j in range(30):
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
    for i in tqdm(range(len(berger_kepler))): # 100
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
    berger_kepler_all.to_csv(path+'galactic-occurrence/systems/berger_kepler_planets_'+str(j)+'.csv', index=False)
"""

"""
Assign galactic heights, transit status, and detected planets for each system.
"""
physical_planet_occurrences = []
detected_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
for i in range(3):
    # read in non-exploded generated system data, which includes non-planet hosts
    berger_kepler_all = pd.read_csv(path+'galactic-occurrence/systems/berger_kepler_planets_'+str(i)+'.csv')
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
    berger_kepler1 = berger_kepler_all.loc[berger_kepler_all['height'] < 150]
    berger_kepler2 = berger_kepler_all.loc[(berger_kepler_all['height'] >= 150) & (berger_kepler_all['height'] < 250)]
    berger_kepler3 = berger_kepler_all.loc[(berger_kepler_all['height'] >= 250) & (berger_kepler_all['height'] < 400)]
    berger_kepler4 = berger_kepler_all.loc[(berger_kepler_all['height'] >= 400) & (berger_kepler_all['height'] < 650)]
    berger_kepler5 = berger_kepler_all.loc[berger_kepler_all['height'] >= 650]

    # bin planet hosts by galactic height
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas'])

    berger_kepler_planets1 = berger_kepler_planets.loc[berger_kepler_planets['height'] < 150]
    berger_kepler_planets2 = berger_kepler_planets.loc[(berger_kepler_planets['height'] >= 150) & (berger_kepler_planets['height'] < 250)]
    berger_kepler_planets3 = berger_kepler_planets.loc[(berger_kepler_planets['height'] >= 250) & (berger_kepler_planets['height'] < 400)]
    berger_kepler_planets4 = berger_kepler_planets.loc[(berger_kepler_planets['height'] >= 400) & (berger_kepler_planets['height'] < 650)]
    berger_kepler_planets5 = berger_kepler_planets.loc[berger_kepler_planets['height'] >= 650]
    berger_kepler_planets_counts = np.array([len(berger_kepler_planets1), len(berger_kepler_planets2), len(berger_kepler_planets3), len(berger_kepler_planets4), len(berger_kepler_planets5)])
    print("physical planet occurrences: ", berger_kepler_planets_counts)

    planet_occurrence1 = len(berger_kepler_planets1)/len(np.unique(berger_kepler1['kepid']))
    planet_occurrence2 = len(berger_kepler_planets2)/len(np.unique(berger_kepler2['kepid']))
    planet_occurrence3 = len(berger_kepler_planets3)/len(np.unique(berger_kepler3['kepid']))
    planet_occurrence4 = len(berger_kepler_planets4)/len(np.unique(berger_kepler4['kepid']))
    planet_occurrence5 = len(berger_kepler_planets5)/len(np.unique(berger_kepler5['kepid']))
    physical_planet_occurrence = np.array([planet_occurrence1, planet_occurrence2, planet_occurrence3, planet_occurrence4, planet_occurrence5])
    print("physical planet occurrence rates, per 100 stars: ", 100*physical_planet_occurrence)
    physical_planet_occurrences.append(100*physical_planet_occurrence)

    detected_planet_occurrences = []
    transit_multiplicities = []
    geom_transit_multiplicities = []

    for i in range(10):    

        berger_kepler_planets_temp = berger_kepler_planets

        ### Simulate detections from these synthetic systems
        prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(berger_kepler_planets_temp.periods, 
                                        berger_kepler_planets_temp.stellar_radius, berger_kepler_planets_temp.planet_radii,
                                        berger_kepler_planets_temp.eccs, 
                                        berger_kepler_planets_temp.mutual_incls, 
                                        berger_kepler_planets_temp.omegas, berger_kepler_planets_temp.stellar_mass,
                                        berger_kepler_planets_temp.rrmscdpp06p0, angle_flag=True) 
        print("prob detections: ", prob_detections)
        print("S/N: ", sn)

        """
        # Try the ExoMULT way (https://github.com/jonzink/ExoMult/blob/master/ScalingK2VIII/ExoMult.py)        
        completeness = probability_detection(berger_kepler_planets_temp.periods, 
                                        berger_kepler_planets_temp.stellar_radius, 2.*np.ones(len(berger_kepler_planets_temp)), # eventually I will draw planet radii
                                        berger_kepler_planets_temp.eccs, 
                                        berger_kepler_planets_temp.mutual_incls, 
                                        berger_kepler_planets_temp.omegas, berger_kepler_planets_temp.stellar_mass,
                                        berger_kepler_planets_temp.rrmscdpp06p0, angle_flag=True)
        """

        berger_kepler_planets_temp['transit_status'] = transit_statuses[0]
        berger_kepler_planets_temp['prob_detections'] = prob_detections[0]
        berger_kepler_planets_temp['sn'] = sn
        berger_kepler_planets_temp['geom_transit_status'] = geom_transit_statuses

        # isolate transiting planets
        berger_kepler_transiters = berger_kepler_planets_temp.loc[berger_kepler_planets_temp['transit_status']==1]

        # read out detected yields
        #berger_kepler_transiters.to_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv', index=False)

        # Read in pre-generated population
        #transiters_berger_kepler = pd.read_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv')

        ### Calculate transit multiplicity and other Population-wide demographics
        #simulate_helpers.collect_galactic(berger_kepler_planets)

        # compute transit multiplicity 
        transit_multiplicity = berger_kepler_transiters.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
        transit_multiplicity = transit_multiplicity.to_list()
        transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
        transit_multiplicities_all.append(transit_multiplicity)

        # also calculate the geometric transit multiplicity
        geom_transiters_berger_kepler = berger_kepler_planets_temp.loc[berger_kepler_planets_temp['geom_transit_status']==1]
        geom_transit_multiplicity = geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid
        geom_transit_multiplicity = geom_transit_multiplicity.to_list()
        geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k
        geom_transit_multiplicities_all.append(geom_transit_multiplicity)

        # calculate logLs 
        #logL = collect_simulations.better_loglike(transit_multiplicity, k)
        #logL_score = collect_simulations.better_loglike(transit_multiplicity, k_score)
        #logL_fpp = collect_simulations.better_loglike(transit_multiplicity, k_fpp)

        """
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

        # bin detected planet hosts by galactic height
        berger_kepler_transiters1 = berger_kepler_transiters.loc[berger_kepler_transiters['height'] < 150]
        berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] >= 150) & (berger_kepler_transiters['height'] < 250)]
        berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] >= 250) & (berger_kepler_transiters['height'] < 400)]
        berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] >= 400) & (berger_kepler_transiters['height'] < 650)]
        berger_kepler_transiters5 = berger_kepler_transiters.loc[berger_kepler_transiters['height'] >= 650]

        detected_planet_occurrence1 = len(berger_kepler_transiters1)/len(np.unique(berger_kepler1['kepid']))
        detected_planet_occurrence2 = len(berger_kepler_transiters2)/len(np.unique(berger_kepler2['kepid']))
        detected_planet_occurrence3 = len(berger_kepler_transiters3)/len(np.unique(berger_kepler3['kepid']))
        detected_planet_occurrence4 = len(berger_kepler_transiters4)/len(np.unique(berger_kepler4['kepid']))
        detected_planet_occurrence5 = len(berger_kepler_transiters5)/len(np.unique(berger_kepler5['kepid']))
        detected_planet_occurrence = np.array([detected_planet_occurrence1, detected_planet_occurrence2, detected_planet_occurrence3, detected_planet_occurrence4, detected_planet_occurrence5])
        print("detected planet occurrences, per 100 stars: ", 100*detected_planet_occurrence)

        detected_planet_occurrences_all.append(detected_planet_occurrence)

    #transit_multiplicities_all.append(transit_multiplicities)
    #geom_transit_multiplicities_all.append(geom_transit_multiplicities)
    #detected_planet_occurrences_all.append(detected_planet_occurrences)

#detected_planet_occurrences_all = 100 * np.array(detected_planet_occurrences_all)

mean_transit_multiplicities = np.mean(transit_multiplicities_all, axis=0)
print("mean transit multiplicities across all systems: ", mean_transit_multiplicities)
mean_geom_transit_multiplicities = np.mean(geom_transit_multiplicities_all, axis=0)
mean_detected_planet_occurrences = np.mean(detected_planet_occurrences_all, axis=0)
print("scale heights: ", zink_kepler['scale_height'])
print("mean detected planet occurrence: ", mean_detected_planet_occurrences)
mean_physical_planet_occurrences = np.mean(physical_planet_occurrences, axis=0)
print("mean physical planet occurrence: ", mean_physical_planet_occurrences)
yerr = np.std(physical_planet_occurrences, axis=0)
print("std of transit multiplicities per bin: ", yerr)

#print(100.*np.max(detected_planet_occurrences_all, axis=0))
#print(100.*np.min(detected_planet_occurrences_all, axis=0))

"""
Fit a line through these points to get best-fit slope for comparison with Zink+ 2023
"""

def linear_model(x, yerr, y=None):
    # These are the parameters that we're fitting and we're required to define explicit
    # priors using distributions from the numpyro.distributions module.
    theta = numpyro.sample("theta", dist.Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0, 1))

    # Transformed parameters (and other things!) can be tracked during sampling using
    # "deterministics" as follows:
    m = numpyro.deterministic("m", jnp.tan(theta))
    b = numpyro.deterministic("b", b_perp / jnp.cos(theta))

    # Then we specify the sampling distribution for the data, or the likelihood function.
    # Here we're using a numpyro.plate to indicate that the data are independent. This
    # isn't actually necessary here and we could have equivalently omitted the plate since
    # the Normal distribution can already handle vector-valued inputs. But, it's good to
    # get into the habit of using plates because some inference algorithms or distributions
    # can take advantage of knowing this structure.
    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(m * x + b, yerr), obs=y)


# Our inference procedure is identical to the one above.
sampler = infer.MCMC(
    infer.NUTS(linear_model),
    num_warmup=2000,
    num_samples=8000,
    num_chains=2,
    progress_bar=True,
)
#print(np.array(zink_kepler['scale_height']), mean_physical_planet_occurrences)

sampler.run(jax.random.PRNGKey(0), np.array(zink_kepler['scale_height']), yerr, y=mean_physical_planet_occurrences)

inf_data = az.from_numpyro(sampler)

print(len(inf_data.posterior.data_vars['m'][0].values))
quit()


print(az.summary(inf_data))
m = inf_data.posterior.data_vars['m'].mean().values
b = inf_data.posterior.data_vars['b'].mean().values
print("slope: ", m)
print("intercept: ", b)

m_std = inf_data.posterior.data_vars['m'].std().values
b_std = inf_data.posterior.data_vars['b'].std().values
print("slope spread: ", m_std)
print("intercept spread: ", b_std)

m_max = inf_data.posterior.data_vars['m'].max().values
b_max = inf_data.posterior.data_vars['b'].max().values
print("slope max: ", m_std)
print("intercept max: ", b_std)



"""
Plot occurrence vs galactic height, as well as transit multiplicity detected yield
"""
#plt.errorbar(x=zink_k2['scale_height'], y=zink_k2['occurrence'], yerr=(zink_k2['occurrence_err1'], zink_k2['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='K2', alpha=0.5)
plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Kepler', alpha=0.5)
plt.scatter(x=zink_kepler['scale_height'], y=100.*physical_planet_occurrence, color='red', label='model [actual]')
plt.fill_between(x=zink_kepler['scale_height'], y1=100.*np.max(detected_planet_occurrences_all, axis=0), y2=100*np.min(detected_planet_occurrences_all, axis=0), color='green', alpha=0.3, label='model [detected]')
plt.plot([100, 1000], b + m * np.array([100, 1000]), label='best-fit', alpha=0.3)
for models in range(9):
    plt.plot([100, 1000], b + m * np.array([100, 1000]), alpha=0.3)


    

plt.xlim([100, 1000])
plt.ylim([6, 100])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("galactic scale height [pc]")
plt.ylabel("planets per 100 stars")
plt.title('f=0.4 if <=2 Gyr; f=0.05 if >2 Gyr')
plt.legend()
plt.tight_layout()
#plt.savefig(path+'galactic-occurrence/plots/test_model_vs_zink6.png')
plt.show()

quit()

# also plot transit multiplicity against Kepler yield, as a diagnostic
print(np.max(transit_multiplicities_all, axis=0))

plt.fill_between(np.arange(7)[1:], np.max(transit_multiplicities, axis=0), np.min(transit_multiplicities, axis=0), color='#03acb1', alpha=0.3, label='model detected') 
#plt.fill_between(np.arange(7)[1:], np.max(geom_transit_multiplicities, axis=0), np.min(geom_transit_multiplicities, axis=0), color='green', alpha=0.3, label='model geometric transit') 
plt.scatter(np.arange(7)[1:], k, color='r', marker='*', s=20, label='Kepler yield (koi_disposition)')
plt.xlabel('number of planets')
plt.ylabel('number of stars')
plt.legend()
plt.savefig(path+'galactic-occurrence/plots/transit-multiplicity6.png', facecolor='white', bbox_inches='tight')
plt.show()