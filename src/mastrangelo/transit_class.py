##################################################
### Functions for the physical pipeline ##########
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
from simulate_helpers import *
import matplotlib.pyplot as plt
import timeit 
from datetime import datetime
import json
from tqdm import tqdm 

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import simulate_helpers
import simulate_transit
#from simulate_helpers import * 
#from simulate_transit import * 

#input_path = '/blue/sarahballard/c.lam/sculpting2/' # HPG
#output_path = '/blue/sarahballard/c.lam/sculpting2/mastrangelo/' # HPG
path = '/Users/chrislam/Desktop/mastrangelo/' 
#berger_kepler = pd.read_csv(input_path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
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

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

class Population: 
    """

    Functions that make populations of planetary systems

    Attributes:
    - kepids [jnp.array]: input Kepler IDs
    - ages [jnp.array]: input stellar ages [Gyr]

    Output:
    - Population object, which is itself populated by Stars objects

    """

    def __init__(
        self, kepids, ages, threshold, frac1, frac2, **kwargs 
    ):
        self.kepids = jnp.array(kepids)
        self.ages = jnp.array(ages)
        self.threshold = threshold
        self.frac1 = frac1
        self.frac2 = frac2
        self.children = []

    def add_child(self, child): 
        self.children.append(child)

    def serialize(self):
        s = {}
        for child in self.children:
            #s[child.kepid.astype(str)] = child.serialize()
            s[child.kepid.astype(str)] = child.reprJSON()
        return s
    
    def sculpting(self, df, m, b, cutoff, bootstrap): # adapted from Ballard et al in prep, log version
        """ 
        Calculate the probability of system being intact vs disrupted, based on its age and the sculpting law.

        Input:
        - df: DataFrame of stars, with age column "iso_age"
        - m: sculpting law slope [dex]
        - b: sculpting law initial intact probability
        - cutoff: sculpting law turnoff time [yrs]
        - bootstrap: do we draw probability of intactness based on iso_age (no bootstrap) or age (bootstrap)

        Output:
        - df: same as input, but with new column called prob_intact

        """
        
        if bootstrap == False:
            df['prob_intact'] = np.where(
                    ((df['iso_age']*1e9 > 1e8) & (df['iso_age']*1e9 <= cutoff)), b+m*(np.log10(df['iso_age']*1e9)-8), np.where(
                        df['iso_age']*1e9 > cutoff, b+m*(np.log10(cutoff)-8), b))

        elif bootstrap == True:
            df['prob_intact'] = np.where(
                    ((df['age']*1e9 > 1e8) & (df['age']*1e9 <= cutoff)), b+m*(np.log10(df['age']*1e9)-8), np.where(
                        df['age']*1e9 > cutoff, b+m*(np.log10(cutoff)-8), b))

        # handle impossible probabilities
        df['prob_intact'] = np.where(
            df['prob_intact']<0, 0, np.where(
                df['prob_intact']>1, 1, df['prob_intact']))
                
        return df

    def galactic_occurrence_step(self, threshold, frac1, frac2):
        """
        Calculate the probability of system having planets, based on its age and three free parameters
        
        Input:
        - threshold: age beyond which probability of hosting a planet is frac2, versus frac1, in Gyr [float]
        - frac1: planet host fraction among systems younger than threshold [float]
        - frac2: planet host fraction among systems older than threshold [float]

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        ages = self.ages
        host_frac = jnp.where(ages <= threshold, frac1, frac2)

        return host_frac
        
    
class Star:
    """

    Functions that make planetary systems, at the per-star level

    Attributes, first four from Berger+ 2020 sample:
    - kepid: Kepler identifier
    - age: drawn stellar age, in Gyr
    - stellar_radius: drawn stellar radius, in Solar radii
    - stellar_mass: drawn stellar mass, in Solar masses
    - rrmscdpp06p0: 6-hr-window CDPP noise [ppm]
    - frac_host: calculated fraction of planet hosts 
    - height: galactic scale height [pc]

    Output:
    - Star object, which is populated by Planets

    """

    def __init__(
        self, kepid, age, stellar_radius, stellar_mass, rrmscdpp06p0, frac_host, height, **kwargs 
    ):

        self.kepid = kepid
        self.age = age
        self.stellar_radius = stellar_radius
        self.stellar_mass = stellar_mass
        self.rrmscdpp06p0 = rrmscdpp06p0
        self.frac_host = frac_host
        self.height = height
        
        #self.midplane = jax.random.uniform(key, minval=-np.pi/2, maxval=np.pi/2)
        self.midplane = np.random.uniform(low=-np.pi/2, high=np.pi/2) # JAX, but I need to figure out how to properly randomly draw

        # prescription for planet-making
        prob_intact = 0.18 + 0.1 * jax.random.normal(key) # from Lam & Ballard 2024; out of planet hosts
        self.prob_intact = prob_intact

        p = simulate_helpers.assign_status(self.frac_host, self.prob_intact)
        self.status = np.random.choice(['no-planet', 'intact', 'disrupted'], p=p)
        #self.intact_flag = assign_flag(key, self.prob_intact, self.frac_host)

        # assign system-level inclination spread based on intact flag
        self.sigma_incl = jnp.where(self.status=='intact', jnp.pi/90, jnp.pi/22.5) # no-planets will also have disrupted spread; need to figure out nested wheres in JAX

        # assign number of planets per system based on intact flag
        self.num_planets = simulate_helpers.assign_num_planets(self.status)

        """
        # populate the Planets here upon initialization
        for i in range(self.num_planets):
            planet = Planet(self.midplane, self.intact_flag, self.sigma_incl)
            self.planets.append(planet)
        """
        if self.num_planets!=None:
            # assign planet planet periods from loguniform distribution from 2 to 300 days
            # Sheila has code to factor in Hill radii for more realistic modeling
            self.periods = jnp.array(loguniform.rvs(2, 300, size=self.num_planets))

            # draw planet radii
            self.planet_radii = simulate_helpers.draw_planet_radii(self.periods)

            # draw inclinations from Gaussian distribution centered on midplane (invariable plane)        
            mu = self.midplane
            sigma = self.sigma_incl
            #self.incls = mu + sigma * jax.random.normal(key, shape=(self.num_planets,)) # JAX, but I need to figure out how to properly randomly draw
            self.incls = np.random.normal(loc=mu, scale=sigma, size=self.num_planets)
            
            # obtain mutual inclinations for plotting to compare {e, i} distributions
            self.mutual_incls = self.midplane - self.incls

            # draw eccentricity
            if (model_flag=='limbach-hybrid') | (model_flag=='limbach'):
                # for drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
                #limbach = pd.read_csv(input_path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
                limbach = pd.read_csv(path+'data/limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
                self.eccs = simulate_helpers.draw_eccentricity_van_eylen_vectorized(model_flag, self.num_planets, limbach)
            else:
                self.eccs = simulate_helpers.draw_eccentricity_van_eylen_vectorized(model_flag, self.num_planets)

            # draw longitudes of periastron
            #self.omegas = jax.random.uniform(key, shape=(self.num_planets,), minval=0, maxval=2*jnp.pi) # JAX, but I need to figure out how to properly randomly draw
            self.omegas = np.random.uniform(low=0, high=2*np.pi, size=self.num_planets)

            # turn to comma-delimited lists for ease of reading in later
            self.incls = self.incls.tolist()
            self.periods = self.periods.tolist()
            self.planet_radii = self.planet_radii.tolist()
            self.mutual_incls = self.mutual_incls.tolist()
            self.eccs = self.eccs.tolist()
            self.omegas = self.omegas.tolist()
            
        else:
            self.periods = None
            self.planet_radii = None
            self.incls = None
            self.mutual_incls = None
            self.eccs = None
            self.omegas = None

    def assign_num_planets(x):
        """
        Based on the status (no planet, dynamically cool, dynamically hot), assign the number of planets

        Input:
        - status: output of assign_status [str]
        
        Output:
        - num_planet: number of planets in the system [int]

        """

        if x=='intact':
            return jax.random.choice([5, 6])
        elif x=='disrupted':
            return jax.random.choice([1, 2])
        elif x=='no-planets':
            return 0

    def reprJSON(self):
        return dict(kepid=self.kepid, age=self.age, frac_host=self.frac_host, midplane=self.midplane, prob_intact=self.prob_intact,
        status=self.status, sigma_incl=self.sigma_incl, num_planets=self.num_planets, periods=self.periods, planet_radii=self.planet_radii, incls=self.incls, 
        mutual_incls=self.mutual_incls, eccs=self.eccs, omegas=self.omegas)  


class Planet:
    """
    Do I need Planets to be a class? For now, no.
    """

    def __init__(
        self, **kwargs 
    ):

        # draw period from loguniform distribution from 2 to 300 days
        self.periods = df.num_planets.apply(lambda x: np.array(loguniform.rvs(2, 300, size=x)))

        # calculate mutual inclination
        self.incl = jaxnp.random.normal(midplane, sigma, 1)
        self.mutual_incl = Star.midplane - self.incl

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

