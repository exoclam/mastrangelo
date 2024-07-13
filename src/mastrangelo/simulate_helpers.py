##################################################
### Helper functions only ########################
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from scipy.interpolate import CubicSpline
from math import lgamma
import jax
import jax.numpy as jnp
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

#path = '/blue/sarahballard/c.lam/sculpting2/'
path = '/Users/chrislam/Desktop/mastrangelo/' # new computer has different username

"""
# Create sample bank of eccentricities to draw from so that you don't call np.searchsorted a bajillion times
# For drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
limbach = pd.read_csv(path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
values = np.random.rand(10000)
value_bins1 = np.searchsorted(limbach['1'], values)
value_bins2 = np.searchsorted(limbach['2'], values)
value_bins5 = np.searchsorted(limbach['5'], values)
value_bins6 = np.searchsorted(limbach['6'], values)
"""

### helper conversion functions
def p_to_a(P, star_mass):
    """
    Newton's full version of Kepler's Third Law, assuming planet mass m --> 0
    Params: 
    - P: days
    - star_mass: Solar masses
    """

    P = P*86400 # days to seconds
    star_mass = star_mass*1.989e30 # solar mass to kg
    a_in_meters = (((P**2) * 6.67e-11 * star_mass)/(4*np.pi**2))**(1./3) # SI units(!)
    a = a_in_meters/(1.496e11) # meters to AU
    return a # in AU

def solar_radius_to_au(radius):
    return 0.00465047*radius

def earth_radius_to_au(radius):
    return 4.26352e-5*radius

def earth_mass_to_cgs(mass):
    return mass*5.974e27 # grams

def solar_mass_to_cgs(mass):
    return mass*1.989e33 # grams

def au_to_cgs(distance):
    return distance*1.496e13 # cm


### helper main functions
def compute_prob(x, m, b, cutoff): # adapted from Ballard et al in prep, log version
    # calculate probability of intact vs disrupted
    
    x = x*1e9
    if x <= 1e8: # we don't care about (nor do we have) systems before 1e8 years
        y = b 

    elif (x > 1e8) & (x <= cutoff): # pre-cutoff regime
        #print(np.log10(x_elt), m, b)
        y = b + m*(np.log10(x)-8)

    elif x > cutoff: # if star is older than cutoff, use P(intact) at cutoff time
        y = b + m*(np.log10(cutoff)-8)

    if y < 0: # handle negative probabilities
        y = 0
    elif y > 1: # handle probabilities greater than 1
        y = 1
            
    return y

def compute_prob_vectorized(df, m, b, cutoff, bootstrap): # adapted from Ballard et al in prep, log version
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

def assign_intact_flag(x):
    # call np.random.choice() for df.apply for model_vectorized()
    # x: df.prod_intact

    return np.random.choice(['intact', 'disrupted'], p=[x, 1-x])

def assign_flag(key, prob_intact, frac_host):
    """
    Assign intact vs disrupted vs no planets

    Inputs:
    - prob_intact: fraction of dynamically cool systems (4-13% for Sun-like Kepler stars)
    - frac_host: fraction of systems with planets

    Output: 
    - status: intact vs disrupted vs no planets [string]
    """

    status = np.random.choice(a=['no-planets', 'intact', 'disrupted'], p=[1-frac_host, frac_host*prob_intact, frac_host*(1-prob_intact)])
    return status

def assign_status(frac_host, prob_intact):
        """
        Label system as having no planet, dynamically cool system, or dynamically hot system.

        Inputs: 
        - frac_host: planet host fraction [float]
        - prob_intact: fraction of dynamically cool systems [float]

        Output:
        - p: list of probabilities of a system being each of the three potential statuses

        """
        
        p = [1-frac_host, frac_host*prob_intact, frac_host*(1-prob_intact)]
        p = np.asarray(p).astype('float64')
        p = p / np.sum(p)

        return p  

def assign_num_planets(x):
    # # call np.random.choice() for df.apply for model_vectorized()
    # x: df.intact_flag
    if x=='intact':
        return np.random.choice([5, 6])
    elif x=='disrupted':
        return np.random.choice([1, 2])
    elif x=='no-planets':
        return 0

def draw_inclinations_vectorized(midplane, sigma, num_planets):
    # call np.random.normal() for df.apply for model_vectorized()
    # x: 
    return np.random.normal(midplane, sigma, num_planets)

def redundancy_check(m, b, cutoff):
    # skip simulations if cutoff occurs more than once after probability has reached zero (use the first one for all)
    # also don't vary cutoffs if m is flat   

    y = b + m*(np.log10(cutoff)-8)
    if y < 0:
        return False
    elif m==0:
        return False
    else:
        return True

### helper physical transit functions
def calculate_eccentricity_limbach(multiplicity):
    """
    Draw eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    Params: multiplicity of system (int)
    Returns: np.array of eccentricity values with length==multiplicity
    """
    # for drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    limbach = pd.read_csv(path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator

    values = np.random.rand(multiplicity) # draw an eccentricity per planet
    if multiplicity==1:
        value_bins = np.searchsorted(limbach['1'], values) # return positions in cdf vector where random values should go
    elif multiplicity==2:
        value_bins = np.searchsorted(limbach['2'], values) # return positions in cdf vector where random values should go
    elif multiplicity==5:
        value_bins = np.searchsorted(limbach['5'], values) # return positions in cdf vector where random values should go
    elif multiplicity==6:
        value_bins = np.searchsorted(limbach['6'], values) # return positions in cdf vector where random values should go
    random_from_cdf = np.logspace(-2,0,101)[value_bins] # select x_d positions based on these random positions
    
    return random_from_cdf

def calculate_eccentricity_limbach_vectorized(multiplicity, limbach):
    """
    Draw eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    Params: multiplicity of system (array of ints); limbach (DataFrame of Limbach & Turner 2014 CDFs)
    Returns: np.array of eccentricity values with length==multiplicity
    """
    if multiplicity > 0:
        values = np.random.rand(multiplicity) # draw an eccentricity per planet

        if multiplicity==1:
            value_bins = np.searchsorted(limbach['1'], values) # return positions in cdf vector where random values should go
        elif multiplicity==2:
            value_bins = np.searchsorted(limbach['2'], values) # return positions in cdf vector where random values should go
        elif multiplicity==5:
            value_bins = np.searchsorted(limbach['5'], values) # return positions in cdf vector where random values should go
        elif multiplicity==6:
            value_bins = np.searchsorted(limbach['6'], values) # return positions in cdf vector where random values should go

        random_from_cdf = np.logspace(-2,0,101)[value_bins] # select x_d positions based on these random positions
        #df['ecc'] = df.apply(lambda x: np.logspace(-2,0,101)[value_bins]) # select x_d positions based on these random positions

        return random_from_cdf
        
    elif multiplicity == 0:
        return np.array([])

def calculate_eccentricity_limbach_vectorized_bank(multiplicity):
    """
    Draw eccentricities using test bank of Limbach & Turner 2014 CDFs
    Params: multiplicity of system (array of ints)
    Returns: np.array of eccentricity values with length==multiplicity
    (under permanent construction)
    """

    if multiplicity==1:
        ecc_indices = np.random.choice(value_bins1, 1)
    elif multiplicity==2:
        ecc_indices = np.random.choice(value_bins2, 2)
    elif multiplicity==5:
        ecc_indices = np.random.choice(value_bins5, 5)
    elif multiplicity==6:
        ecc_indices = np.random.choice(value_bins6, 6)
    
    random_from_cdf = np.logspace(-2,0,101)[ecc_indices] # select x_d positions based on these random positions
    #df['ecc'] = df.apply(lambda x: np.logspace(-2,0,101)[value_bins]) # select x_d positions based on these random positions
    
    return random_from_cdf

def draw_eccentricity_van_eylen_vectorized(model_flag, num_planets, *args):
    # *args is optional parameter of the limbach DataFrame for certain model_flags

    if model_flag=='limbach-hybrid':
        limbach = args[0]
        sigma_rayleigh = 0.26
        #draws = np.where(num_planets > 1, calculate_eccentricity_limbach(num_planets), np.random.rayleigh(sigma_rayleigh, num_planets))
        draws = np.where(num_planets > 2, calculate_eccentricity_limbach_vectorized(num_planets, limbach), np.random.rayleigh(sigma_rayleigh, num_planets))
        #draws = np.where(num_planets > 1, calculate_eccentricity_limbach_vectorized_bank(num_planets), np.random.rayleigh(sigma_rayleigh, num_planets))

    elif model_flag=='rayleigh':
        sigma_single = 0.24
        sigma_multi = 0.061
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi

        draws= np.random.rayleigh(sigma, num_planets)

    return draws

def draw_eccentricity_van_eylen(model_flag, num_planets):
    """
    Draw eccentricities per the four models of Van Eylen et al 2018 (https://arxiv.org/pdf/1807.00549.pdf)
    Params: flag (string) saying which of the four models; num_planets (int)
    Returns: list eccentricity per planet in the system
    """
    if model_flag=='rayleigh':
        sigma_single = 0.24
        sigma_multi = 0.061
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi
            
        draw = np.random.rayleigh(sigma, num_planets)

    elif model_flag=='half-Gaussian':
        sigma_single = 0.32
        sigma_multi = 0.083
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi
            
        draw = np.random.normal(0, sigma, num_planets)
        if any(d < 0 for d in draw): # put the 'half' in half-Gaussian by redrawing if any draw element is negative
            draw = draw_eccentricity_van_eylen('half-Gaussian', num_planets)
        
    elif model_flag=='beta':
        a_single = 1.58
        b_single = 4.4
        a_multi = 1.52
        b_multi = 29
        
        # errors for pymc3 implementation of eccentricity draws, should I wish to get fancy.
        # I currently do not wish to get fancy.
        a_single_err1 = 0.59
        a_single_err2 = 0.93
        b_single_err1 = 1.8
        b_single_err2 = 2.2
        a_multi_err1 = 0.5
        a_multi_err2 = 0.85
        b_multi_err1 = 9
        b_multi_err2 = 17
        
        if num_planets==1:
            a = a_single
            b = b_single
        elif num_planets>1:
            a = a_multi
            b = b_multi
        
        draw = np.random.beta(a, b, num_planets)
        
    elif model_flag=='mixed':
        sigma_half_gauss = 0.049
        sigma_rayleigh = 0.26
        f_single = 0.76
        f_multi = 0.08
        
        if num_planets==1:
            draw = np.random.rayleigh(sigma_rayleigh, num_planets)
        elif num_planets>1:
            draw = np.random.normal(0, sigma_half_gauss, num_planets)
            if any(d < 0 for d in draw): # redraw if any planet's eccentricity is negative
                draw = draw_eccentricity_van_eylen('mixed', num_planets)
                
    elif model_flag=='limbach-hybrid':
        """
        Testing something for Sarah: use Rayleigh for intact and Limbach for disrupted. 
        """
        sigma_rayleigh = 0.26
        #print(num_planets)
        if num_planets==1:
            draw = np.random.rayleigh(sigma_rayleigh, num_planets)
        elif num_planets>1:
            draw = calculate_eccentricity_limbach(num_planets)

    elif model_flag=='limbach': # OG way of drawing eccentricities, from Limbach & Turner 2014
        draw = calculate_eccentricity_limbach(num_planets)

    elif model_flag=='absurd': # sanity check
        if num_planets==1:
            draw = np.random.uniform(0.09, 0.11, num_planets)
        elif num_planets>1:
            draw = np.random.uniform(0.009, 0.011, num_planets)
            
    return draw

def calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag): # following Winn 2010 Eqn 7
    """
    angle_flag: True means indexed at 0; False means indexed at pi/2
    """
    star_radius = solar_radius_to_au(star_radius)
    if angle_flag==True:
        factor1 = (a * np.cos(np.pi/2 - incl))/star_radius  # again, we're indexed at 0 rather than pi/2
    elif angle_flag==False: # if indexed at pi/2
        factor1 = (a * np.cos(incl))/star_radius 
    factor2 = (1-e**2)/(1+e*np.sin(omega)) # leave this alone, right? Bc everyone always assumes omega=pi/2?
    
    return factor1 * factor2

def calculate_impact_parameter_vectorized(star_radius, a, e, incl, omega, angle_flag): # following Winn 2010 Eqn 7
    """
    angle_flag: True means indexed at 0; False means indexed at pi/2
    """
    star_radius = solar_radius_to_au(star_radius)
    if angle_flag==True:
        factor1_temp = np.pi/2 - incl
        factor1 = (a * factor1_temp.apply(lambda x: np.cos(x)))/star_radius  # again, we're indexed at 0 rather than pi/2
    elif angle_flag==False: # if indexed at pi/2
        factor1_temp = incl
        factor1 = (a * factor1_temp.apply(lambda x: np.cos(x)))/star_radius 
    factor2 = (1-e**2)/(1+e*omega.apply(lambda x: np.sin(x))) # leave this alone, right? Bc everyone always assumes omega=pi/2?
    
    return factor1 * factor2

def calculate_transit_duration_vectorized(P, r_star, r_planet, b, a, inc, e, omega, angle_flag): # Winn 2011s Eqn 14 & 16
    #print("take 1: ", r_planet/r_star)
    #print("take 2: ", (1+(r_planet/r_star))**2 - b**2)
    
    arg1_temp = (1+(r_planet/r_star))**2 - b**2
    arg1 = arg1_temp.apply(lambda x: np.sqrt(x))
    if angle_flag==True:
        arg2_temp = np.pi/2 - inc
        arg2 = (r_star / a) * (arg1 / arg2_temp.apply(lambda x: np.sin(x))) # account for us being indexed at 0 rather than pi/2
    elif angle_flag==False:
        arg2_temp = inc
        arg2 = (r_star / a) * (arg1 / arg2_temp.apply(lambda x: np.sin(x))) # assuming indexed at pi/2
    arg3_temp = 1-e**2
    arg3 = arg3_temp.apply(lambda x: np.sqrt(x))/(1+e*omega.apply(lambda x: np.sin(x))) # eccentricity factor from Eqn 16 of Winn 2011
    #print("Winn args: ", arg1, arg2, arg3)
    
    return (P / np.pi) * arg2.apply(lambda x: np.arcsin(x)) * arg3

def calculate_transit_duration(P, r_star, r_planet, b, a, inc, e, omega, angle_flag): # Winn 2011s Eqn 14 & 16
    #print("take 1: ", r_planet/r_star)
    #print("take 2: ", (1+(r_planet/r_star))**2 - b**2)
    
    arg1 = np.sqrt((1+(r_planet/r_star))**2 - b**2)
    if angle_flag==True:
        arg2 = (r_star / a) * (arg1 / np.sin(np.pi/2 - inc)) # account for us being indexed at 0 rather than pi/2
    elif angle_flag==False:
        arg2 = (r_star / a) * (arg1 / np.sin(inc)) # assuming indexed at pi/2
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16 of Winn 2011
    #print("Winn args: ", arg1, arg2, arg3)
    
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_paw(P, star_radius, planet_radius, b, a, incl, e, omega): # Paul Anthony Wilson website: https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    arg1 = np.sqrt((star_radius+planet_radius)**2 - (b*star_radius)**2) 
    arg2 = arg1/a
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    #print("PAW args: ", arg1, arg2, arg3)
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_he(P, star_radius, planet_radius, a, e, omega): # from Matthias He: https://github.com/ExoJulia/SysSimExClusters/tree/master/src
    arg1 = (P/np.pi) * (star_radius/a)
    arg2 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    arg2 = 1
    return arg1 * arg2

def calculate_amd(m_pks, m_star, a_ks, e_ks, i_ks, multiplicity):
    ### 
    # calculate angular momentum deficit following Eqn 13 from Milholland et al 2021
    ###
    amd = []
    for i in range(multiplicity):
        lambda_k = m_pks[i] * np.sqrt(G*m_star*a_ks[i])
        second_term = 1 - (np.sqrt(1 - (e_ks[i])**2))*np.cos(i_ks[i])
        amd.append(lambda_k * second_term)
        
    return np.sum(amd)

### helper transit detection functions
def calculate_sn(P, rp, rs, cdpp, tdur, unit_test_flag=False): 
    """
    Calculate S/N per planet using Eqn 4 in Christiansen et al 2012: https://arxiv.org/pdf/1208.0595.pdf
    
    Params: P (days); rp (Earth radii); rs (Solar radii); cdpp (ppm); tdur (days)
    
    Returns: S/N
    """
    tobs = 365*3.5 # days; time spanned observing the target; set to 3.5 years, or the length of Kepler mission
    f0 = 0.92 # fraction of time spent actually observing and not doing spacecraft things
    tcdpp = 0.25 # days; using CDPP for 6 hour transit durations; could change to be more like Earth transiting Sun?
    rp = solar_radius_to_au(rp) # earth_radius_to_au when not using Matthias's test set
    rs = solar_radius_to_au(rs)
    #print(P, rp, rs, cdpp, tdur)
    
    factor1 = np.sqrt(tobs*f0/np.array(P)) # this is the number of transits
    delta = 1e6*(rp/rs)**2 # convert from parts per unit to ppm
    cdpp_eff = cdpp * np.sqrt(tcdpp/tdur)
    #print("CDPP ingredients: ", cdpp, tcdpp, tdur)
    factor2 = delta/cdpp_eff
    sn = factor1 * factor2
    
    if unit_test_flag==True:
        if np.isnan(sn)==True:
            sn = 0
        return sn
    else:
        sn = sn.fillna(0)
        return sn

def calculate_sn_vectorized(P, rp, rs, cdpp, tdur, unit_test_flag=False): 
    
    """
    Calculate S/N per planet using Eqn 4 in Christiansen et al 2012: https://arxiv.org/pdf/1208.0595.pdf
    
    Params: P (days); rp (Earth radii); rs (Solar radii); cdpp (ppm); tdur (days)
    
    Returns: S/N
    """

    tobs = 365*3.5 # days; time spanned observing the target; set to 3.5 years, or the length of Kepler mission
    f0 = 0.92 # fraction of time spent actually observing and not doing spacecraft things
    tcdpp = 0.25 # days; using CDPP for 6 hour transit durations; could change to be more like Earth transiting Sun?
    rp = solar_radius_to_au(rp) # earth_radius_to_au when not using Matthias's test set
    rs = solar_radius_to_au(rs)

    factor1 = tobs*f0/P.apply(lambda x: np.sqrt(x)) # this is the number of transits
    delta = 1e6*(rp/rs)**2 # convert from parts per unit to ppm
    cdpp_eff = cdpp * tcdpp/tdur.apply(lambda x: np.sqrt(x))

    factor2 = delta/cdpp_eff
    sn = factor1 * factor2

    """
    NEED TO KEEP NANS ACTUALLY FOR FREE INFORMATION ON GEOMETRIC TRANSITS
    if unit_test_flag==True:
        if np.isnan(sn)==True:
            sn = 0
        return sn
    else:
        sn = sn.fillna(0)
        return sn
    """

    return sn

def draw_cdpp(star_radius, df):
    df = df.loc[(df.st_radius<star_radius+0.15)&(df.st_radius>star_radius-0.15)]
    #print("df length: ", len(df))
    cdpp = np.random.choice(df.rrmscdpp06p0)
    return cdpp

def draw_cdpp_array(star_radius, df):
    # calculate CDPP by drawing from Kepler dataset relation with star radius
    cdpp = [draw_cdpp(sr, berger_kepler) for sr in star_radius]
    return cdpp

def make_pdf_rows(x, mode, err1, err2):
    """
    Row-wise likelihood of asymmetric uncertainty, using Eqn 6 from https://iopscience.iop.org/article/10.3847/1538-3881/abd93f
    Less efficient, but row-wise in order to troubleshoot. 
    Takes in scalar values for mode, err1, and err2, not arrays.

    Input:
    - x: np.linspace(0.5, 10, 100); just something to undergird the PDF and represent stellar ages
    - mode: mean age; peak of asymmetric PDF
    - err1: + uncertainty
    - err2: - uncertainty (note: must be positive)

    Output:
    - pdf: asymmetric PDF

    """
    
    factor1 = 1/np.sqrt(2*np.pi*err1**2) 
    beta = err1/err2
    gamma = (err1 - np.abs(err2))/(err1 * np.abs(err2))
    
    factor2_arg_a = np.log(1+gamma*(x - mode))
    factor2_arg_b = np.log(beta)
    factor2_arg = factor2_arg_a/factor2_arg_b
    factor2 = np.exp(-0.5*(factor2_arg)**2)

    if np.isnan(factor1*factor2).all():
        print(mode, err1, err2)

    out = factor1 * factor2
    out[~np.isfinite(out)] = 0.0
    
    return out

def draw_asymmetrically(df, mode_name, err1_name, err2_name, drawn):
    """
    Draw stellar properties with asymmetric errors. 
    This is the generalized version of draw_star_ages(), below
    
    Inputs:
    - df: berger_kepler [Pandas DataFrame]
    - mode_name: name of mode column [string]
    - err1_name: name of err1 column [string]
    - err2_name: name of err2 column [string]
    - drawn: name of new column [string]

    Output:
    - df: berger_kepler_df, now with new column with drawn parameter, "drawn" [Pandas DataFrame]
    """

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])
    
    x = np.linspace(0.5, 10, 100)
    modes = np.ones(len(uniques))
    for i in range(len(uniques)):
        mode = uniques.iloc[i][mode_name]
        err1 = uniques.iloc[i][err1_name]
        err2 = np.abs(uniques.iloc[i][err2_name])

        # symmetric uncertainties
        if err1==err2:
            draw = 0
            while mode <= 0: # make sure the draw is positive
                draw = np.around(np.random.normal(mode, err1), 2)

        # asymmetric uncertainties
        elif err1!=err2:
            pdf = make_pdf_rows(x, mode, err1, err2)
            pdf = pdf/np.sum(pdf)

            try:
                draw = 0
                while draw <= 0: # make sure the draw is positive
                    draw = np.around(np.random.choice(x, p=pdf), 2)
            except:
                print(i, pdf, mode, err1, err2)
                break
        
        modes[i] = mode

    df[drawn] = modes

    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df[drawn] = df[drawn].fillna(method='ffill')
    
    return df

def draw_star_ages(df):
    """
    Draw star's age, taking into account asymmetric age errors. Enriches input DataFrame.
    """

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    x = np.linspace(0.5, 10, 100)
    ages = np.ones(len(uniques))
    for i in range(len(uniques)):
        mode = uniques.iloc[i].iso_age
        err1 = uniques.iloc[i].iso_age_err1
        err2 = np.abs(uniques.iloc[i].iso_age_err2)

        # symmetric age uncertainties
        if err1==err2:
            age = 0
            while age <= 0: # make sure the age is positive
                age = np.around(np.random.normal(mode, err1), 2)

        # asymmetric uncertainties
        elif err1!=err2:
            pdf = make_pdf_rows(x, mode, err1, err2)
            pdf = pdf/np.sum(pdf)

            try:
                age = 0
                while age <= 0: # make sure the age is positive
                    age = np.around(np.random.choice(x, p=pdf), 2)
            except:
                print(i, pdf, mode, err1, err2)
                break
        
        ages[i] = age

    df['age'] = ages

    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['age'] = df.age.fillna(method='ffill')

    return df 

def draw_star(df):
    """
    Draw star's age, metallicity, and effective temperature based on given errors. Enrich input DataFrame.
    """

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    # draw age
    uniques['iso_age_err'] = 0.5 * (uniques.iso_age_err1 + np.abs(uniques.iso_age_err2)) # old way

    uniques['age'] =  np.random.normal(uniques.iso_age, uniques.iso_age_err)

    # draw metallicity...if I do feh instead of iso_feh, do I get a lot of NaNs??
    uniques['feh_err'] = 0.5 * (uniques.feh_err1 + np.abs(uniques.feh_err2))
    uniques['feh'] = np.random.normal(uniques.feh_x, uniques.feh_err)

    # draw Teff
    uniques['iso_teff_err'] = 0.5 * (uniques.iso_teff_err1 + np.abs(uniques.iso_teff_err2))
    uniques['teff'] = np.random.normal(uniques.iso_teff, uniques.iso_teff_err)

    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['age'] = df.age.fillna(method='ffill')
    df['feh'] = df.feh.fillna(method='ffill')

    return df

def draw_radii():
    """
    There are many ways to get radii (Exoplanet Archive, Petigura+, etc.). 
    Require user to specify where to get them from, so they critically reckon with their choices. 
    """

    return

def galactic_occurrence_step(ages, threshold, frac1, frac2):
        """
        Calculate the probability of system having planets, based on its age and three free parameters
        
        Input:
        - ages: stellar ages, in Gyr [float]
        - threshold: age beyond which probability of hosting a planet is frac2, versus frac1, in Gyr [float]
        - frac1: planet host fraction among systems younger than threshold [float]
        - frac2: planet host fraction among systems older than threshold [float]

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        host_frac = jnp.where(ages <= threshold, frac1, frac2)

        return host_frac

def hill_radius(a, m):
    """
    Calculate Hill radii among all consecutive pairs

    Input: 
    - a: list of planets' semi-major axes
    - m: list of planets' masses

    Output:
    - check: does this set of planets contain no planet that falls within another's Hill radius? [bool]
    """

    return check

def draw_planet_radii(periods):
    """
    Draw planet radii following Zink+ 2023: https://iopscience.iop.org/article/10.3847/1538-3881/acd24c#ajacd24cs4. 
    This starts with a simple power law.
    Then we partition using the Van Eylen+ 2018 planet radius-period relation: https://academic.oup.com/mnras/article/479/4/4786/5050069
    We do not yet offer size ordering within systems. 

    Input: 
    - periods: planet periods [np.array of floats]

    Output: 
    - radii: planet radii [np.array of floats]
    """

    # mise en place
    # are these going to be Super-Earths (se) or Sub-Neptunes (sn)? Zink+ 2023 says their occurrence rates overlap, so in a world with only small planets, let's say 50-50
    # should I assume Peas in a Pod? Should I allow for that? 
    se_or_sn = np.random.choice(['se','sn'], size=len(periods), p=[0.5, 0.5])
    se_grid = np.linspace(1.2, 2., 100)
    sn_grid = np.linspace(2., 4., 100)
    period_grid = np.logspace(np.log10(3), np.log10(200), 100)
    m_grid = np.linspace(-0.5, 0.5, 100) # for radius gap
    a_grid = np.linspace(np.log10(1), np.log10(4), 100) # for radius gap

    # draw alpha, which is the power relation between radius and the PDF of the radius distribution
    alpha_se = np.random.normal(-1., 0.2)
    alpha_sn = np.random.normal(-1.5, 0.1)

    # generate PDF for radius, and then normalize to sum to 1
    q_se = se_grid**alpha_se
    q_se = q_se/np.sum(q_se)

    q_sn = sn_grid**alpha_sn
    q_sn = q_sn/np.sum(q_sn)
    
    # draw initial radii
    radii = [] # this will be flexible if I want to break Peas in a Pod later
    radii_se = np.random.choice(se_grid, size=len(se_or_sn[se_or_sn == 'se']), p=q_se)
    radii_sn = np.random.choice(sn_grid, size=len(se_or_sn[se_or_sn == 'sn']), p=q_sn)
    radii.append(radii_se)
    radii.append(radii_sn)
    radii = np.concatenate([x.ravel() for x in radii])

    """
    Cull radius valley planets. Maybe 5%? 
    """
    # draw m, which is the power law slope between radius and period
    radius_valley_m_pdf = make_pdf_rows(m_grid, -0.09, 0.02, 0.04)
    radius_valley_m_pdf = radius_valley_m_pdf/np.sum(radius_valley_m_pdf)
    m = np.around(np.random.choice(m_grid, p=radius_valley_m_pdf), 2)

    # draw upper and lower envelope y-intercepts 
    radius_valley_a_upper_pdf = make_pdf_rows(a_grid, 0.44, 0.04, 0.03)
    radius_valley_a_upper_pdf = radius_valley_a_upper_pdf/np.sum(radius_valley_a_upper_pdf)
    a_upper = np.around(np.random.choice(a_grid, p=radius_valley_a_upper_pdf), 2)

    radius_valley_a_lower_pdf = make_pdf_rows(a_grid, 0.29, 0.04, 0.03)
    radius_valley_a_lower_pdf = radius_valley_a_lower_pdf/np.sum(radius_valley_a_lower_pdf)
    a_lower = np.around(np.random.choice(a_grid, p=radius_valley_a_lower_pdf), 2)

    # is it in the radius valley?
    upper = 10**(m * np.log10(periods) + a_upper)
    lower = 10**(m * np.log10(periods) + a_lower)

    # resample 95% of the time if a planet lands in here
    for index, radius in enumerate(radii):

        # only for gap planets
        while (radius >= lower[index]) & (radius <= upper[index]):

            # 5% chance of grace 
            grace = np.random.choice(['accept gap', 'reject gap'], p=[0.05, 0.95])
            if grace=='accept gap':
                break
            elif grace=='reject gap':
                pass

            # resample until planet is out of gap
            if radius <= 2.:
                radii[index] = np.random.choice(se_grid, size=1, p=q_se)[0]
            elif (radius > 2.) & (radius <= 4.):
                radii[index] = np.random.choice(sn_grid, size=1, p=q_sn)[0]
            #print("try: ", radii[index], lower[index], upper[index])

    """
    # plot to check if radius sampling and valley are working as expected; feed in: stats.loguniform.rvs(2, 300, size=1000)
    df = pd.DataFrame({'p': periods, 'r': radii, 'upper': upper, 'lower': lower})
    df = df.sort_values(by='p')
    plt.scatter(df.p, df.r, s=10)
    plt.xscale('log')
    plt.plot(df.p, df.upper, label='upper', color='k')
    plt.plot(df.p, df.lower, label='lower', color='r')
    plt.xlabel('period [days]')
    plt.ylabel('radius [$R_{\oplus}$]')
    plt.legend(bbox_to_anchor=(1., 1.05))
    plt.tight_layout()
    plt.savefig(path+'radius-v-period.png')
    plt.show()
    """

    return radii

def collect_galactic(df):
    """
    Compute geometric and detected transit multiplicities, as well as other population-wide statistics, like fraction of planet hosts and fraction of intact systems
    
    This is different from collect() in the dynamical sculpting case because the models there affected intact_frac, not frac_host. 
    So transit multiplicity doesn't assume a constant f. In fact, it already contains it. 

    Also, the scope of this function is per Star, not per Population. 

    Input:
    - df: Pandas DataFrame of planet-hosting systems, with rows broken down by planet, not star

    Outputs:
    - transit_multiplicity_among_planet_hosts: 
    """

    # isolate transiting planets
    transiters_berger_kepler = df.loc[df['transit_status']==1]

    # compute transit multiplicity 
    transit_multiplicity_among_planet_hosts = transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
    transit_multiplicity_among_planet_hosts = transit_multiplicity_among_planet_hosts.to_list()
    transit_multiplicity_among_planet_hosts += [0.] * (6 - len(transit_multiplicity_among_planet_hosts)) # pad with zeros to match length of k

    # also calculate the geometric transit multiplicity
    geom_transiters_berger_kepler = df.loc[df['geom_transit_status']==1]
    geom_transit_multiplicity = geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid
    geom_transit_multiplicity = geom_transit_multiplicity.to_list()
    geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k

    # calculate logLs 
    logL = better_loglike(transit_multiplicity, k)
    logL_score = better_loglike(transit_multiplicity, k_score)
    logL_fpp = better_loglike(transit_multiplicity, k_fpp)

    # get intact and disrupted fractions (combine them later to get fraction of systems w/o planets)
    intact = df.loc[df.intact_flag=='intact']
    disrupted = df.loc[df.intact_flag=='disrupted']
    intact_frac = len(intact.kepid.unique())/len(df.kepid.unique())
    disrupted_frac = len(disrupted.kepid.unique())/len(df.kepid.unique())

    return transit_multiplicity_among_planet_hosts, geom_transit_multiplicity_among_planet_hosts, intact_fracs, disrupted_fracs, logLs, logLs_score, logLs_fpp

def normalize(array):
    array = 10**array # exponentiate first!!
    sum = np.sum(array)
    
    return array/sum

def draw_galactic_heights(df):

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    # Read in stellar density vs galactic height curves from Ma+ 2017 Fig 2: https://academic.oup.com/mnras/article/467/2/2430/2966031. I converted the image to data using PlotDigitizer.
    data1 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-0-2Gyr.csv', header=None, 
                    names=['height','density'])
    data2 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-2-4Gyr.csv', header=None, # Ma_midplaneheight_2Gyr_4Gyr.txt
                    names=['height','density'])
    data3 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-4-6Gyr.csv', header=None, 
                    names=['height','density'])
    data4 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-6-8Gyr.csv', header=None, 
                    names=['height','density'])
    data5 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-8-moreGyr.csv', header=None, 
                    names=['height','density'])

    data1_height = np.linspace(0, np.array(data1['height'])[-1], 100)
    data2_height = np.linspace(0, np.array(data2['height'])[-1], 100)
    data3_height = np.linspace(0, np.array(data3['height'])[-1], 100)
    data4_height = np.linspace(0, np.array(data4['height'])[-1], 100)
    data5_height = np.linspace(0, np.array(data5['height'])[-1], 100)

    # interpolate for finer galactic height draws
    cs1 = CubicSpline(data1['height'], data1['density'])
    cs2 = CubicSpline(data2['height'], data2['density'])
    cs3 = CubicSpline(data3['height'], data3['density'])
    cs4 = CubicSpline(data4['height'], data4['density'])
    cs5 = CubicSpline(data5['height'], data5['density'])

    data1_density = cs1(data1_height)
    data2_density = cs2(data2_height)
    data3_density = cs3(data3_height)
    data4_density = cs4(data4_height)
    data5_density = cs5(data5_height)

    data1_density = normalize(data1_density) 
    data2_density = normalize(data2_density) 
    data3_density = normalize(data3_density) 
    data4_density = normalize(data4_density) 
    data5_density = normalize(data5_density) 

    # draw heights based on stellar ages
    heights = []
    for i in range(len(df)):
        if uniques['age'][i] <= 2.:
            height = np.random.choice(data1_height, p=data1_density)
        elif (uniques['age'][i] > 2.) & (uniques['age'][i] <= 4.):
            height = np.random.choice(data2_height, p=data2_density)
        elif (uniques['age'][i] > 4.) & (uniques['age'][i] <= 6.):
            height = np.random.choice(data3_height, p=data3_density)
        elif (uniques['age'][i] > 6.) & (uniques['age'][i] <= 8.):
            height = np.random.choice(data4_height, p=data4_density)
        elif uniques['age'][i] > 8.:
            height = np.random.choice(data5_height, p=data5_density)
        heights.append(height)

    df['height'] = np.array(heights) * 1000

     # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['height'] = df.height.fillna(method='ffill')

    return df 