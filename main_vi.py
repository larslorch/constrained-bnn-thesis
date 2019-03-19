# from operator import mul, add
# from functools import reduce
# import scipy.stats as stat
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm
# import os

# import pprint
# import inspect

import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from plot import *
from utils import *
from exe_vi import *
from bbb import bayes_by_backprop_variational_inference
from bnn import make_BNN
from pp_violation import compute_posterior_predictive_violation_vi


'''

TODOs

- find out why posterior predictive has no confidence intervals at all... hyperparam tuning

- compute violation function passed from bbb function for trianing evaluation
- constraint heatmap
- multithreading via pytorch

'''


def main_vi(all_experiments):

    for id, experiment in enumerate(all_experiments):

        print('Experiment {} / {}.'.format(id + 1, len(all_experiments)))

        if experiment['vi']['load_saved']:
            current_directory = make_unique_dir(experiment, method='vi')
            
            best, params, training_evaluations = joblib.load(
                'experiment_results/' + experiment['vi']['load_from'] + '/vi/' + experiment['title'] + '_data.pkl')

            '''BNN '''
            architecture = experiment['nn']['architecture']
            nonlinearity = experiment['nn']['nonlinearity']
            prior_ds = experiment['nn']['prior_ds']
            noise_ds = experiment['data']['noise_ds']
            batch_size = experiment['vi']['batch_size']
            num_batches = int(torch.ceil(torch.tensor(X.shape[0] / batch_size))) if batch_size else 1

            _, forward, _ = \
                make_BNN(layer_sizes=architecture,
                            prior_ds=prior_ds,
                            noise_ds=noise_ds,
                            nonlinearity=nonlinearity,
                            num_batches=num_batches)

            def sample_q(samples, params):
                if experiment['vi']['alg'] == 'bbb':
                    mean, log_std = params[:, 0], params[:, 1]
                    weights = mean + torch.randn(samples,
                                                mean.shape[0]) * log_std.exp()
                    return weights
                if experiment['vi']['alg'] == 'npv':
                    N = params.shape[0]
                    k = ds.Categorical(probs=torch.ones(N)).sample(torch.Size([samples]))
                    means, log_stds = params[:, 1:], params[:, 0]
                    means, log_stds = means[k], log_stds[k]
                    num_weights = means.shape[1]
                    covs = torch.zeros(samples, num_weights, num_weights)
                    I = torch.eye(num_weights)
                    for j in range(samples):
                        covs[j] = log_stds[j].exp().pow(2) * I
                    return ds.MultivariateNormal(means, covs).sample()

        else:
            # run (both regular and constrained experiment)
            results, funcs, current_directory = run_experiment(experiment)
            best, params, training_evaluations = results
            forward = funcs['forward']
            sample_q = funcs['sample_q']
            
        

        # compute posterior predictive violation
        pcvs = compute_posterior_predictive_violation_vi(
            params, forward, sample_q, experiment)

        '''Plotting'''
        # posterior predictive
        param = params[best]
        function_samples = 200
        samples = sample_q(function_samples, param)

        plot_posterior_predictive(
            samples, forward, experiment, current_directory, method='vi', j='best')

        for j in range(len(params)):
            plot_posterior_predictive(
                samples, forward, experiment, current_directory, method='vi', j=j)

        '''Print table info to out'''
        X_v_id = experiment['data']['X_v_id']
        Y_v_id = experiment['data']['Y_v_id']
        X_v_ood = experiment['data']['X_v_ood']
        Y_v_ood = experiment['data']['Y_v_ood']

        rmse_id = compute_rmse(X_v_id, Y_v_id, samples, forward)
        rmse_ood = compute_rmse(X_v_ood, Y_v_ood, samples, forward)
        held_out_ll_id = held_out_loglikelihood(X_v_id, Y_v_id, samples, forward)
        held_out_ll_ood = held_out_loglikelihood(X_v_ood, Y_v_ood, samples, forward)

        print('PCV: {}\nHeld-out LogLik ID: {}\nRMSE ID: {}\nHeld-out LogLik OOD: {}\nRMSE OOD: {}'.format(
            pcvs[best], held_out_ll_id, rmse_id, held_out_ll_ood, rmse_ood))


        '''Log results'''
        # training evaluations
        # plot_training_evaluation(
        #     experiment, training_evaluations, current_directory)


        # log_results(experiment, training_evaluations, violations,
        #             best, current_directory, descr)


if __name__ == '__main__':

    print('Run main() function in experiment file.')
