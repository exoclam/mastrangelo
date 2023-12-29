from jax.config import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
import jax
from jax import random, tree_map, vmap, jit, numpy as jnp
from jax import vmap

from jaxns import ExactNestedSampler
from jaxns import Model
from jaxns import PriorModelGen, Prior
from jaxns import TerminationCondition
from jaxns.types import float_type
from jaxns.internals.log_semiring import LogSpace
#from jaxns import bruteforce_evidence

from helpers import bruteforce_evidence, period_ratios
from math import lgamma

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

tfpd = tfp.distributions
random = np.random.default_rng(seed=42)

# period ratio distribution between 1.4 and 1.6 from the Kepler sample
k = np.array([6., 9., 22., 10.])

# set path for input
path = '/Users/chris/Desktop/mastrangelo/'
kepler_planet_enriched = pd.read_csv(path+'data/pnum_plus_cands_fgk.csv')

# reset path for outputs
path = path+'period-ratios/'

# the dataset won't change for this experiment

### example test

tfpd = tfp.distributions
tfpk = tfp.math.psd_kernels

N = 100
M = N // 2
Xstar = jnp.linspace(-2., 2., N)[:, None]
X = Xstar[:M]

true_uncert = 0.2

spectral_params_true = dict(
    logits=jnp.asarray([0., 0.]),
    locs=jnp.asarray([1. / 1, 1. / 0.5]),
    scales=jnp.asarray([1. / 4., 1 / 0.8])
)
prior_cov = tfpk.SpectralMixture(**spectral_params_true).matrix(Xstar, Xstar) + 1e-12 * jnp.eye(N)

v = jnp.linspace(0., 2., 100)
kern = tfpk.SpectralMixture(**spectral_params_true).matrix(jnp.zeros((1, 1)), v[:, None])

Y = jnp.linalg.cholesky(prior_cov) @ jax.random.normal(jax.random.PRNGKey(42), shape=(N,))
Y_obs = Y[:M] + true_uncert * jax.random.normal(jax.random.PRNGKey(1), shape=(M,))
# Y = jnp.cos(jnp.pi*2. * X[:,0]/2) + jnp.exp(- X[:,0]/2) * jnp.sin(jnp.pi*2. * X[:,0]/3)

from jaxns import Prior, Model, ExactNestedSampler, ForcedIdentifiability

kernel = tfpk.SpectralMixture

def log_normal2(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    # U, S, Vh = jnp.linalg.svd(cov)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))  # jnp.sum(jnp.log(S))#
    dx = x - mean
    dx = solve_triangular(L, dx, lower=True)
    # U S Vh V 1/S Uh
    # pinv = (Vh.T.conj() * jnp.where(S!=0., jnp.reciprocal(S), 0.)) @ U.T.conj()
    maha = dx @ dx  # dx @ pinv @ dx#solve_triangular(L, dx, lower=True)
    log_likelihood = -0.5 * x.size * jnp.log(2. * jnp.pi) - log_det - 0.5 * maha
    return log_likelihood

def log_likelihood2(uncert, kernel_params):
    """
    P(Y|sigma, half_width) = N[Y, f, K]
    Args:
        uncert: noise
        kernel_params: dict of kernel parameters
        
    Returns:
        log likelihood
    """
    print("kernel: ", kernel_params)
    print("kernel double asterisk: ", **kernel_params)
    K = tfpk.SpectralMixture(**kernel_params).matrix(X, X)
    data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
    mu = jnp.zeros_like(Y_obs)
    return log_normal(Y_obs, mu, K + data_cov)


# Build the model
n_components = 2


def prior_model2():
    smallest_dt = jnp.min(jnp.diff(jnp.sort(X, axis=0), axis=0))  # d
    largest_dt = jnp.max(X, axis=0) - jnp.min(X, axis=0)  # d
    period = yield ForcedIdentifiability(n=2, low=smallest_dt, high=largest_dt, name='period')  # n, d
    locs = yield Prior(1. / period, name='locs')  # n, d
    max_bandwidth = (4. * period)  # n, d
    min_bandwidth = smallest_dt * 2
    bandwidth = yield Prior(
        tfpd.Uniform(min_bandwidth * jnp.ones_like(max_bandwidth), max_bandwidth * jnp.ones_like(max_bandwidth)),
        name='bandwidth')  # n, d
    scales = yield Prior(1. / bandwidth, name='scales')
    logits = yield Prior(tfpd.Normal(0. * jnp.ones(n_components), 1. * jnp.ones(n_components)), name='logits')  # n
    kernel_params = dict(locs=locs,
                         scales=scales,
                         logits=logits)
    uncert = yield Prior(tfpd.HalfNormal(0.2), name='uncert')
    return uncert, kernel_params


model = Model(prior_model=prior_model2, log_likelihood=log_likelihood2)

quit()

#### resume programming



@jit
def generate_model(timescale, end, formation, commensurability):
    """
    Reassign period ratios for multi-planet systems in the Kepler sample

    Inputs: 
    - timescale: time at which period ratio shifts from commensurability [Gyr]
    - end: final period ratio after shift
    - formation: probability that planet formation mechanism resulted in random distribution of periods (in-situ), not resonant (migration)
    - threshold: commensurability, eg. 3:2, 2:1
    """
    print("params: ", timescale, end, formation)
    
    fadsfadf

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
    keep = period_ratios(df, commensurability)

    ### THIS PART IS MODEL-DEPENDENT
    print("formation: ", formation)
    ### choose planet formation mechanism based on "formation" probability
    formation = 0.4
    formation_flag = jax.random.choice(key=jax.random.PRNGKey(42), a=np.array([0, 1]), p=np.array([formation, 1-formation]))
    print("formation_flag: ", formation_flag)

    ### hacking a LogUniform distribution into JAX lol
    timescale = 10**timescale
    print("timescale: ", timescale)

    @jit
    def migration_fn():
        return jax.lax.cond()

    # new way
    #jax.lax.cond(formation_flag == 0, keep.ratio = np.random.uniform(len(keep)), 
    #    migration_fn, x)
    
    # old way
    if formation_flag==0: # in-situ
        keep.ratio = np.random.uniform(len(keep))
    elif formation_flag==1: # migration
        keep.loc[keep.age < timescale].ratio = commensurability
        keep.loc[keep.age >= timescale].ratio = end

    # bin with histogram for "model" yield
    bins = np.arange(1.4, 1.6, 0.05)
    y, x, _ = plt.hist(keep.ratio, bins=bins)

    lam = np.array(y)
    print("model: ", (timescale, end, formation), "yield: ", lam)

    return lam

@jit
def log_likelihood(kernel_params, threshold=1.5):
    print("PARAMS: ", *kernel_params)
    adfadf

    lam = generate_model(timescale, end, formation, threshold)

    logL = []
    for i in range(len(lam)):
        if lam[i]==0:
            term3 = -lgamma(k[i]+1)
            term2 = -lam[i]
            term1 = 0
            logL.append(term1+term2+term3)

        else:
            #print("k: ", k.astype(float))
            #print("k element: ", k[i])

            term3 = -lgamma(k[i]+1)
            term2 = -lam[i]
            term1 = k[i]*np.log(lam[i])
            logL.append(term1+term2+term3)

    return np.sum(logL)

def prior_model() -> PriorModelGen:
    timescale = yield Prior(tfpd.Uniform(low=8., high=10.), name='timescale')
    end = yield Prior(tfpd.Uniform(low=1.5, high=1.6), name='end')
    formation = yield Prior(tfpd.Uniform(low=0., high=1.), name='formation')

    kernel_params = dict(timescale=timescale,
                         end=end,
                         formation=formation)

    return kernel_params #timescale, end, formation

model = Model(prior_model=prior_model,
              log_likelihood=log_likelihood)

"""
log_Z_true = bruteforce_evidence(model=model, S=250)
print(f"True log(Z)={log_Z_true}")

u_vec = jnp.linspace(0., 1., 250)
args = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * model.U_ndims, indexing='ij')], axis=-1)

# The `prepare_func_args(log_likelihood)` turns the log_likelihood into a function that nicely accepts **kwargs
lik = vmap(model.forward)(args).reshape((u_vec.size, u_vec.size))

#plt.imshow(jnp.exp(lik), origin='lower', extent=(-4, 4, -4, 4), cmap='jet')
#plt.colorbar()
#plt.show()
"""

# Create the nested sampler class. In this case without any tuning.
exact_ns = ExactNestedSampler(model=model, num_live_points=200, num_parallel_samplers=1, max_samples=1e4)

termination_reason, state = exact_ns(jax.random.PRNGKey(42),
                                     term_cond=TerminationCondition(live_evidence_frac=1e-4))
results = exact_ns.to_results(state, termination_reason)

# We can use the summary utility to display results
exact_ns.summary(results)

# We plot useful diagnostics and a distribution cornerplot
exact_ns.plot_diagnostics(results)
exact_ns.plot_cornerplot(results)