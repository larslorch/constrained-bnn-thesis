import os
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from plot import *
from utils import *
from bbb import bayes_by_backprop_variational_inference
from bnn import make_BNN
from npv import nonparametric_variational_inference


'''
Computes mass of posterior predictive in constrained region
(i.e. independent of constraint function parameters)

This is an interpretable evaluation metric rather than the violation part of the objective.
'''


def compute_posterior_predictive_violation_vi(params, forward, sample_q, experiment):

    S = experiment['vi']['posterior_predictive_analysis']['posterior_samples']
    T = experiment['vi']['posterior_predictive_analysis']['constrained_region_samples_for_pp_violation']

    constrained_region_sampler = experiment['vi']['constrained']['constrained_region_sampler']
    integral_constrained_region = experiment['data']['integral_constrained_input_region']

    constr = experiment['constraints']['constr']

    violations = []

    # for each random restart
    for j, param in enumerate(params):

        '''Collect samples from optimized variational distribution'''
        ws = sample_q(S, param)

        '''Integral of posterior predictive over total constrained region, evaluation metric'''

        # 1 - find random x samples form constrained region (via passed in sampling function)
        #     and approximation of area of constrained x region

        all_mc_points = constrained_region_sampler(T).unsqueeze(-1)

        # 2 - approximate integral using monte carlo

        integral = 0
        all_x_mc_points = []
        all_y_mc_points = []
        mc_points_color = []

        for x_ in all_mc_points:

            # 2.1 - sample ys from p(y' | x', X, Y) using MC samples of W
            ys = forward(ws, x_)

            # 2.2 - approximate mass in constrained region by ys that satisfy constraint
            ys_violated = 0

            for y_ in ys:

                polytopes_violated = []
                for region in constr:

                    polytopes_violated.append(all(
                        [c_x(x_, y_) <= 0 for c_x in region]))

                if any(polytopes_violated):
                    ys_violated += 1

                    all_x_mc_points.append(x_)
                    all_y_mc_points.append(y_)
                    mc_points_color.append('red' if any(
                        polytopes_violated) else 'green')

            mass_violated = ys_violated / ys.shape[0]
            integral += ((1 / T) * mass_violated) * \
                integral_constrained_region

        violations.append(integral)

    return violations



'''HMC version - not evaluating multiple restarts nor able to control the number of samples '''
def compute_posterior_predictive_violation_hmc(samples, forward, experiment):

    T = experiment['vi']['posterior_predictive_analysis']['constrained_region_samples_for_pp_violation']
    constr = experiment['constraints']['constr']
    constrained_region_sampler = experiment['vi']['constrained']['constrained_region_sampler']
    
    # 1D : length of constrained input region; 
    # 2D : area of constrained input region; 
    # etc.
    integral_constrained_input_region = experiment['data']['integral_constrained_input_region']


    '''Integral of posterior predictive over total constrained region, evaluation metric'''

    # 1 - find random x samples form constrained region (via passed in sampling function)
    #     and approximation of area of constrained x region

    all_mc_points = constrained_region_sampler(T).unsqueeze(-1)

    # 2 - approximate integral using monte carlo

    integral = 0
    all_x_mc_points = []
    all_y_mc_points = []

    for x_ in all_mc_points:

        # 2.1 - sample ys from p(y' | x', X, Y) using MC samples of W
        ys = forward(samples, x_)

        # 2.2 - approximate mass in constrained region (given x') by y' that violate constraint
        ys_violated = 0

        for y_ in ys:

            polytopes_violated = []
            for region in constr:

                polytopes_violated.append(all(
                    [c_x(x_, y_) <= 0 for c_x in region]))

            if any(polytopes_violated):
                ys_violated += 1

                all_x_mc_points.append(x_)
                all_y_mc_points.append(y_)

        mass_violated = ys_violated / ys.shape[0]

        # 2.3 - approximate mass in _complete_ constrained input region 
        #       by averaging/summing up slices for every sample of x' 

        integral += mass_violated * \
            ((1 / T) * integral_constrained_input_region)

    return integral
