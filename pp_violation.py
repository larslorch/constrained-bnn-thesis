import os
import joblib
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from plot import *
from utils import *
from plot_CONST import f_prototype
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


'''Compute average metrics at the end of BBB across versions'''

if __name__ == '__main__':

    core = 'tab_4_3_convergence_analysis'
    versions = [
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
    ]
    gammas = [0, 1, 10, 100, 1000, 10000]
 
 
    all_version_evals = []
    all_version_params = []
    for j in range(len(versions)):
        all_evals = []
        all_params = []
        for i in range(len(gammas)):
            file = core + '_{}'.format(gammas[i])
            file_version = file + '_v{}'.format(versions[j][i])
            best, params, training_evaluations = joblib.load(
                'experiment_results/' + file_version + '/vi/' + file + '_data.pkl')
            all_evals.append(training_evaluations)
            all_params.append(params)
        all_version_evals.append(all_evals)
        all_version_params.append(all_params)

    '''Table (assuming BBB)'''

    exp_gam = copy.deepcopy(f_prototype)


    def y_0(x, y):
        return y + 0.5 + 2 * torch.exp(- x.pow(2))


    def y_1(x, y):
        return - y + 0.5 + 2 * torch.exp(- x.pow(2))


    constr = [
        [y_0],
        [y_1]
    ]

    def constrained_region_sampler(s):
        out = ds.Uniform(-6, 6).sample(sample_shape=torch.Size([s, 1]))
        return out

    exp_gam['vi']['run_constrained'] = True
    exp_gam['nn']['architecture'] = [1, 20, 1]
    exp_gam['constraints']['constr'] = constr
    exp_gam['data']['integral_constrained_input_region'] = 12
    exp_gam['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler
    

    num_weights, forward, _ = \
        make_BNN(layer_sizes=exp_gam['nn']['architecture'],
                 prior_ds=exp_gam['nn']['prior_ds'],
                 noise_ds=exp_gam['data']['noise_ds'],
                 nonlinearity=exp_gam['nn']['nonlinearity'],
                 num_batches=0)

    def sample_q(samples, params):
        mean, log_std = params[:, 0], params[:, 1]
        weights = mean + torch.randn(samples,
                                     mean.shape[0]) * log_std.exp()
        return weights


    # table
    elbo = torch.zeros(len(versions), len(gammas))
    pcv = torch.zeros(len(versions), len(gammas))
    ll_id = torch.zeros(len(versions), len(gammas))
    rmse_id = torch.zeros(len(versions), len(gammas))
    ll_ood = torch.zeros(len(versions), len(gammas))
    rmse_ood = torch.zeros(len(versions), len(gammas))

  
    for i in range(len(versions)):
        for j in range(len(gammas)):
            params = all_version_params[i][j]
            eval = all_version_evals[i][j]
            cache = compute_posterior_predictive_violation_vi(
                params, forward, sample_q, exp_gam)
            pcv[i][j] = cache[0]
            elbo[i][j] = eval[0]['elbo'][-1]
            ll_id[i][j] = eval[0]['held_out_ll_indist'][-1]
            rmse_id[i][j] = eval[0]['rmse_id'][-1]
            ll_ood[i][j] = eval[0]['held_out_ll_outofdist'][-1]
            rmse_ood[i][j] = eval[0]['rmse_ood'][-1]
    

    print('Gamma: {}'.format(gammas))
    print('ELBO: {}'.format(elbo.mean(0).numpy()))
    print('pcv: {}'.format(pcv.mean(0).numpy()))
    print('ll_id: {}'.format(ll_id.mean(0).numpy()))
    print('rmse_id: {}'.format(rmse_id.mean(0).numpy()))
    print('ll_ood: {}'.format(ll_ood.mean(0).numpy()))
    print('rmse_ood: {}'.format(rmse_ood.mean(0).numpy()))
