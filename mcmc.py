#!/usr/bin/env python
import numpy as np
from scipy import stats
# MCMC MH sampler code. Lifted from
# https://github.com/tdhopper/mcmc/blob/master/Metropolis-Hastings%20Algorithm.ipynb


def transition(current_state, logpdf, dim):
    transition = stats.multivariate_normal.rvs(size=dim)
    candidate = current_state + transition
    prev_log_likelihood = logpdf(current_state)
    candidate_log_likelihood = logpdf(candidate)
    if np.isinf(candidate_log_likelihood).any():
        return current_state
    diff = candidate_log_likelihood - prev_log_likelihood
    uniform_draw = np.log(stats.uniform(0, 1).rvs())
    return candidate if uniform_draw < diff else current_state

def generate_samples(initial_state, num_iterations, logpdf):
    current_state = initial_state
    
    if isinstance(current_state, float) or isinstance(current_state, int):
        dim = 1 
    else:
        dim = len(current_state)
        
    for i in range(num_iterations):
        current_state = transition(current_state, logpdf, dim)
        yield current_state