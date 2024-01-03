import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import lgamma
from helpers import period_ratios, numpy_callback, times_one
import dynesty
from dynesty import plotting as dyplot

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
kepler_planet_enriched = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv')

# reset path for outputs
path = path+'period-ratios/'

# dynesty initialization prep
rstate = np.random.default_rng(819)
ndim = 3  
cube = np.zeros(ndim)

def prior_transform(cube):
	"""
	
	"""
	cube[0] = 10**np.random.uniform(8,10) 
	cube[1] = np.random.uniform(1.5,1.6)
	cube[2] = np.random.uniform(0,1) 
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
    uniques['iso_age_err'] = 0.5 * (uniques.iso_age_err1 + np.abs(uniques.iso_age_err2))
    uniques['age'] = np.random.normal(uniques.iso_age, uniques.iso_age_err)
    
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

# initialize our nested sampler
sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim, nlive=100,
                               rstate=rstate)

# sample from the target distribution
sampler.run_nested(dlogz=0.1)

res = sampler.results  # grab our results
print('Keys:', res.keys(),'\n')  # print accessible keys
res.summary()  # print a summary

# plot corner plots
# initialize figure
fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]

# plot noiseless run (left)
fg, ax = dyplot.cornerplot(res, color='blue', 
                           show_titles=True, max_n_ticks=3, title_kwargs={'y': 1.05},
                           quantiles=None, fig=(fig, axes[:, :3]))

plt.savefig(path+'corner.png')