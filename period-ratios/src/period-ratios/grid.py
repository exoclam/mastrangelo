import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import lgamma
from helpers import period_ratios

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

# period ratio distribution between 1.4 and 1.6 from the Kepler sample
k = np.array([6., 9., 22., 10.])

# set path for input
path = '/Users/chrislam/Desktop/mastrangelo/'
path = '/home/c.lam/blue/period-ratios/'
output_path = path+'out/'
kepler_planet_enriched = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv')

cube = [0, 0, 0]

def prior_transform(cube, gi_a, gi_b, gi_c):

	"""
	Each model run will use an evenly spaced tuple on a discrete 20 x 10 x 20 grid

    Inputs:
    - gi_a: log timescale at which dynamical sculpting of period ratios occurs
    - gi_b: possible end states of period ratios within the relevant period ratio window
    - gi_c: probability of a system following in-situ formation (random distribution) or migration

    Outputs:
    - cube of populated hyperparameters

	"""

	cube[0] = np.logspace(8, 10, 20)[gi_a]
	cube[1] = np.linspace(1.5, 1.6, 11)[gi_b]
	cube[2] = np.linspace(0, 1, 11)[gi_c]

	return cube

def loglikelihood(cube): # timescale, end, formation

    timescale, end, formation = cube[0], cube[1], cube[2]
    lam = generate_model(timescale, end, formation)

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

def generate_model(timescale, end, formation):
    """
    Reassign period ratios for multi-planet systems in the Kepler sample

    Inputs: 
    - timescale: time at which period ratio shifts from commensurability [Gyr]
    - end: final period ratio after shift
    - formation: probability that planet formation mechanism resulted in random distribution of periods (in-situ), not resonant (migration)
    - threshold: commensurability, eg. 3:2, 2:1
    """
    #print("params: ", timescale, end, formation)

    df = kepler_planet_enriched

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    # draw age
    uniques['age'] = np.random.uniform(uniques.iso_age + uniques.iso_age_err2, uniques.iso_age + uniques.iso_age_err1)
    
    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['age'] = df.age.ffill()

    # keep only planet pairs in relevant period ratio range
    keep = period_ratios(df)

    ### for each system, choose planet formation mechanism based on "formation" probability
    formation_flag = np.random.choice(a=[0,1], size=len(keep), p=[formation, 1-formation])
    keep['formation_flag'] = formation_flag

    # select rows where second column value < threshold; set third column to commensurability
    #keep[keep[:,1] > 4, 2] = 1.5
    in_situ = keep.loc[keep.formation_flag == 0]
    migration = keep.loc[keep.formation_flag == 1]

    # re-assign period ratios
    in_situ.ratio = np.random.uniform(low=1.4, high=1.6, size=len(in_situ))
    migration.loc[migration.age < timescale].ratio = 1.5
    migration.loc[migration.age >= timescale].ratio = end

    # re-combine DataFrames
    keep = pd.concat([in_situ, migration])

    # bin with histogram for "model" yield
    bins = np.arange(1.4, 1.6, 0.05)
    y, x, _ = plt.hist(keep.ratio, bins=bins)

    lam = np.array(y)
    print("model: ", (timescale, end, formation), "yield: ", lam)
    return lam

def main_recovery(cube):
    """
    CREATE 30 REALIZATIONS FOR EACH STAR, USING ERRORS.
    FOR EACH REALIZATION, COMPARE 
    """
    gi_as = []
    gi_bs = []
    models = []
    logLs = []
    for gi_a in range(20):
        print(gi_a)
        for gi_b in range(11):				
            for gi_c in range(11): 

                gi_as.append(gi_a)
                gi_bs.append(gi_b)

                # fetch hyperparams
                cube = prior_transform(cube, gi_m, gi_b, gi_c)
                timescale, end, formation = cube[0], cube[1], cube[2]

                # for each model, draw 30 times and generate yields
                lams = []
                for i in range(30):
                    lam = generate_model(timescale, end, formation)
                    lams.append(lam)

                model = np.mean(lams, axis=1) # element-wise average
                models.append(model)

                # calculate logLs
                logL = better_loglike(model, k)
                logLs.append(logL)

    output = pd.DataFrame({'timescale': gi_as, 'end': gi_bs, 'formation': np.repeat(np.linspace(0, 1, 11)),
        'model': models, 'logL': logLs})
    print(output)

    output.to_csv(output_path+'output.csv', index=False)

    return

main_recovery(cube)
