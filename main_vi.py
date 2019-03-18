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

        # run (both regular and constrained experiment)
        results, funcs, current_directory = run_experiment(experiment)

        best, params, training_evaluations, descr = results
            
        param = params[best]

        # compute posterior predictive violation
        violations = compute_posterior_predictive_violation_vi(
            params, funcs, experiment)

        '''Plotting'''

        # posterior predictive
        forward = funcs['forward']
        sample_q = funcs['sample_q']
        plt_x_domain = experiment['data']['plt_x_domain']
        plt_y_domain = experiment['data']['plt_y_domain']
        X = experiment['data']['X']
        Y = experiment['data']['Y']
        X_plot = experiment['data']['X_plot']
        Y_plot = experiment['data']['Y_plot']
        X_v_id = experiment['data']['X_v_id']
        Y_v_id = experiment['data']['Y_v_id']
        X_v_ood = experiment['data']['X_v_ood']
        Y_v_ood = experiment['data']['Y_v_ood']

        plot_patch = experiment['constraints']['plot_patch']
        plot_between = experiment['constraints']['plot_between']
        size_tup = experiment['data']['plt_size']

        function_samples = 200

        samples = sample_q(function_samples, param)

        plot_posterior_predictive(
            samples, forward, experiment, current_directory, method='vi')

        '''Log results'''
        # training evaluations
        plot_training_evaluation(
            experiment, training_evaluations, current_directory)


        log_results(experiment, training_evaluations, violations,
                    best, current_directory, descr)


if __name__ == '__main__':

    print('Run main() function in experiment file.')
