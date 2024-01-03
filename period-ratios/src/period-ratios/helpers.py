from jax.config import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
import jax
from jax import random, numpy as jnp, tree_map, vmap, jit
from jax import vmap

from jaxns import ExactNestedSampler
from jaxns import Model
from jaxns import PriorModelGen, Prior
from jaxns import TerminationCondition
from jaxns.internals.log_semiring import LogSpace

import itertools
from itertools import combinations

def times_one(x):
    return 1*x
    
### callback tracer
@jit
def numpy_callback(x):
  # Need to forward-declare the shape & dtype of the expected output.
  result_shape = jax.core.ShapedArray(x.shape, x.dtype)
  return jax.pure_callback(np.sin, result_shape, x)

def quotient(a, b):
    return max(a,b)/min(a,b)

def period_ratios(df):
    
    """
    Calculate period ratio distributions for young and old samples at different thresholds
    
    Input: 
    - df: kepler_planet_enriched DataFrame
    
    Returns: all ratios [np.arrays]
    
    """
    
    all_ratios = []
    ages = []
    kepids = []
    for i in np.unique(df.kepid):
        system = df.loc[df.kepid == i]
        
        if len(system) > 1: # if multi

            pairs = itertools.combinations(system.koi_period, r=2)
            
            for pair in pairs:

                all_ratios.append(quotient(*pair))
                ages.append(system.reset_index().age.iloc[0])
                kepids.append(system.reset_index().kepid.iloc[0])

    all_ratios = np.array(all_ratios)
    ages = np.array(ages)
    kepids = np.array(kepids)

    #return kepids, ages, all_ratios 
    #tf.data.Dataset.from_tensor_slices((numeric_features, target))
    all = pd.DataFrame({'kepid': kepids, 'age': ages, 'ratio': all_ratios})

    # keep only rows in relevant period ratio range
    keep = all.loc[(all.ratio <= 1.5+0.1) & (all.ratio >= 1.5-0.1)]
    
    return keep # all_ratios
