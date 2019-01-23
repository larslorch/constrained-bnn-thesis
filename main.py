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
from evaluation import *
from bbb import bayes_by_backprop_variational_inference
from bnn import make_BNN


'''

TODOs

- find out why posterior predictive has no confidence intervals at all... hyperparam tuning

- compute violation function passed from bbb function for trianing evaluation
- constraint heatmap
- multithreading via pytorch

'''


if __name__ == '__main__':

    from evaluation_input_tanh import all_experiments

    for id, experiment in enumerate(all_experiments):

        print('Experiment {} / {}.'.format(id + 1, len(all_experiments)))

        # run (both regular and constrained experiment)
        results, bnn_forward_pass, current_directory = run_experiment(experiment)

        # analysis options
        show_function_samples = experiment['experiment']['show_function_samples']
        show_posterior_predictive = experiment['experiment']['show_posterior_predictive']
        show_plot_training_evaluations = experiment['experiment']['show_plot_training_evaluations']
        show_constraint_function_heatmap = experiment['experiment']['show_constraint_function_heatmap']

        # perform same analysis
        for best, params, training_evaluations, descr in results:
            
            opt_param = params[best]

            # compute posterior predictive violation
            violations = compute_posterior_predictive_violation(
                params, bnn_forward_pass, experiment)           

            '''Plotting'''

            if show_function_samples or show_posterior_predictive or \
               show_plot_training_evaluations or show_constraint_function_heatmap:

                plot_directory = current_directory + '/plots_' + descr
                os.makedirs(plot_directory)

            # sample functions
            if show_function_samples:
                plot_sample_functions(
                    opt_param, bnn_forward_pass, experiment, plot_directory, descr)

            # posterior predictive
            if show_posterior_predictive:
                plot_posterior_predictive(
                    opt_param, bnn_forward_pass, experiment, plot_directory, descr)

            # training evaluations
            if show_plot_training_evaluations:
                plot_training_evaluation(
                    experiment, training_evaluations, plot_directory, descr)

            # constraint function
            if show_constraint_function_heatmap:
                plot_constraint_heatmap(
                    opt_param, bnn_forward_pass, experiment, plot_directory)
                

            '''Log results'''
            log_results(experiment, training_evaluations, violations,
                        best, current_directory, descr)






