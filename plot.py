import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from utils import *

dpi = 400


'''
Classes for plotting constrained regions
'''

class DrawRectangle:

    def __init__(self, bottom_left=(-1, -1), top_right=(1, 1)):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def get(self):
        return patches.Rectangle(self.bottom_left,
                                 self.top_right[0] - self.bottom_left[0],
                                 self.top_right[1] - self.bottom_left[1],
                                 alpha=0.3,
                                 color='red')


'''
Plot for functions sampled from single variational approximation
'''

def plot_sample_functions(param, prediction, experiment, plot_directory, descr):

    ground_truth_f = experiment['data']['ground_truth']
    plt_x_domain = experiment['data']['plt_x_domain']
    plt_y_domain = experiment['data']['plt_y_domain']
    X = experiment['data']['X']
    Y = experiment['data']['Y']
    X_plot = experiment['data']['X_plot']
    constr_plot = experiment['constraints']['plot']

    size_tup = experiment['experiment']['plot_size']

    function_samples = 10

    # can only plot 1D at this point
    if X_plot.shape[1] == 1:

        mean, log_std = param[:, 0], param[:, 1]
        ws = mean + torch.randn(function_samples, param.shape[0]) *  log_std.exp() # torch.log(1.0 + log_std.exp()) 
        samples = prediction(ws, X_plot).squeeze()

        if len(samples.shape) > 2:
            print('Can only plot 1D output.')
            return

        fig, ax = plt.subplots()
        for y in samples:
            ax.plot(X_plot.numpy(), y.numpy())

        for p in constr_plot:
            ax.add_patch(p.get())

        plt.title('Posterior samples using {}'.format(descr))
        plt.plot(X_plot.numpy(), ground_truth_f(X_plot).numpy(),
                color='blue', alpha=0.7, linestyle='--')
        plt.scatter(X.numpy(), Y, color='black', marker='x')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(plt_x_domain)
        plt.ylim(plt_y_domain)

        plt.tight_layout()
        fig = plt.gcf()  # get current figure
        fig.set_size_inches(size_tup)
        plt.savefig(
            plot_directory + '/function_samples_{}.png'.format(descr), format='png', frameon=False, dpi=dpi)
        plt.close('all')


'''
Plot for posterior predictive of single variational approximation
'''

def plot_posterior_predictive(param, prediction, experiment, plot_directory, descr):

    ground_truth_f = experiment['data']['ground_truth']
    plt_x_domain = experiment['data']['plt_x_domain']
    plt_y_domain = experiment['data']['plt_y_domain']
    X = experiment['data']['X']
    Y = experiment['data']['Y']
    X_plot = experiment['data']['X_plot']
    X_v_id = experiment['data']['X_v_id']
    X_v_ood = experiment['data']['X_v_ood']

    Y_v_id = ground_truth_f(X_v_id)
    Y_v_ood = ground_truth_f(X_v_ood)


    constr_plot = experiment['constraints']['plot']
    size_tup = experiment['experiment']['plot_size']

    function_samples = 200

    # can only plot 1D at this point
    if X_plot.shape[1] == 1:

        mean, log_std = param[:, 0], param[:, 1]
        ws = mean + torch.randn(function_samples,
                                param.shape[0]) * log_std.exp() # torch.log(1.0 + log_std.exp())
        samples = prediction(ws, X_plot).squeeze()
        mean = samples.mean(0).squeeze().numpy()
        std = samples.std(0).squeeze().numpy()

        if len(samples.shape) > 2:
            print('Can only plot 1D output.')
            return

        '''Plot'''
        fig, ax = plt.subplots()

        ax.scatter(X.numpy(), Y.numpy(), color='black', marker='x')
        ax.plot(X_plot.numpy(), mean, c='black')
        ax.plot(X_plot.numpy(), ground_truth_f(X_plot).numpy(),
                color='blue', alpha=0.7, linestyle='--')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.fill_between(X_plot.squeeze().numpy(),
                        mean - 1 * std,
                        mean + 1 * std,
                        color='black',
                        alpha=0.3)
        ax.fill_between(X_plot.squeeze().numpy(),
                        mean - 2 * std,
                        mean + 2 * std,
                        color='black',
                        alpha=0.2)

        for p in constr_plot:
            ax.add_patch(p.get())

        ax.set_xlim(plt_x_domain)
        ax.set_ylim(plt_y_domain)

        plt.title('Posterior predictive using {}'.format(descr))

        plt.tight_layout()
        fig = plt.gcf()  # get current figure
        fig.set_size_inches(size_tup)
        plt.savefig(
            plot_directory + '/posterior_predictive_{}.png'.format(descr), format='png', frameon=False, dpi=dpi)
        plt.close('all')

        # same plot but with held-out test data
        
        fig, ax = plt.subplots()

        ax.plot(X_plot.numpy(), mean, c='black')
        ax.plot(X_plot.numpy(), ground_truth_f(X_plot).numpy(),
                color='blue', alpha=0.7, linestyle='--')
        ax.set_xlabel('')
        ax.set_ylabel('y')
        ax.fill_between(X_plot.squeeze().numpy(),
                        mean - 1 * std,
                        mean + 1 * std,
                        color='black',
                        alpha=0.3)
        ax.fill_between(X_plot.squeeze().numpy(),
                        mean - 2 * std,
                        mean + 2 * std,
                        color='black',
                        alpha=0.2)
        for p in constr_plot:
            ax.add_patch(p.get())
        ax.set_xlim(plt_x_domain)
        ax.set_ylim(plt_y_domain)
        ax.scatter(X_v_id.numpy(), Y_v_id.numpy(), color='lawngreen', marker='x')
        ax.scatter(X_v_ood.numpy(), Y_v_ood.numpy(), color='orange', marker='x')

        plt.title('Posterior predictive using {}'.format(descr))

        plt.tight_layout()
        fig = plt.gcf()  # get current figure
        fig.set_size_inches(size_tup)
        plt.savefig(
            plot_directory + '/held_out_data_{}.png'.format(descr), format='png', frameon=False, dpi=dpi)
        plt.close('all')

        
'''
Summary plot of training metrics
'''


def plot_training_evaluation(experiment, training_evaluations, plot_directory, descr):

    size_tup = experiment['experiment']['plot_size']

    objs = [dat['objective'] for dat in training_evaluations]
    elbos = [dat['elbo'] for dat in training_evaluations]
    violations = [dat['violation']
                  for dat in training_evaluations]

    cutoff = 1  # ignore first <cutoff> datapoints

    step = experiment['bbb']['regular']['reporting_every_']
    start = cutoff * step
    end = (len(objs[0])) * step

    for e in objs:
        plt.plot(
            torch.arange(start=start, end=end, step=step).numpy(), e[cutoff:])
    plt.title('Objective function ({})'.format(descr))
    # _, top = plt.ylim() # makes bottom end zero
    # plt.ylim((0, top))

    plt.tight_layout()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(size_tup)
    plt.savefig(
        plot_directory + '/objective_function_{}.png'.format(descr), format='png', frameon=False, dpi=dpi)
    plt.close('all')

    fig, ax = plt.subplots(1, 3, figsize=(
        2 * size_tup[0], size_tup[1]))

    # f, axarr = plt.subplots(2, 2)
    # obj
    for e in objs:
        ax[0].plot(
            torch.arange(start=start, end=end, step=step).numpy(), e[cutoff:])
    ax[0].set_title('Objective function')
    # _, top = ax[0].get_ylim() # makes bottom end zero
    # ax[0].set_ylim((0, top))

    # elbo
    for e in elbos:
        ax[1].plot(
            torch.arange(start=start, end=end, step=step).numpy(), e[cutoff:])
    ax[1].set_title('ELBO')

    # violation
    for e in violations:
        ax[2].plot(
            torch.arange(start=start, end=end, step=step).numpy(), e[cutoff:])
    ax[2].set_title('Constraint violation [%]')
    ax[2].set_ylim((-0.05, 1.05)) # TODO these limits only make sense if it is percentage not the violation term from the objective

    plt.suptitle(descr)
    # plt.tight_layout()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches((2 * size_tup[0], size_tup[1]))
    plt.savefig(
        plot_directory + '/training_{}.png'.format(descr), format='png', frameon=False, dpi=dpi)
    plt.close('all')


'''
Heatmap of constraint function (only if input and output are 1D)
'''

def plot_constraint_heatmap(param, prediction, experiment, plot_directory):

    # TODO

    '''
    ground_truth_f = experiment['data']['ground_truth']
    plt_x_domain = experiment['data']['plt_x_domain']
    plt_y_domain = experiment['data']['plt_y_domain']

    constr_plot = experiment['constraints']['plot']
    size_tup = experiment['experiment']['plot_size']

    size_tup = experiment['experiment']['plot_size']  # width, height

    r = np.linspace(plt_x_domain[0], plt_x_domain[1], num=100)
    s = np.flip(
        np.linspace(plt_y_domain[0], plt_y_domain[1], num=100))
    a = np.zeros((r.shape[0], s.shape[0]))
    for i, x in enumerate(r):
        for j, y in enumerate(s):
            a[i, j] = bbb.violates_general_broadcast(
                x, y, constr, tau=tau_tup)
    # plt.imshow(a.T, cmap='hot', interpolation='nearest',
    #            extent=(plt_x_domain[0], plt_x_domain[1], plt_y_domain[0], plt_y_domain[1]), aspect=4)
    plt.imshow(a.T, cmap='hot', interpolation='nearest')

    # plt.tight_layout()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches((size_tup[0], size_tup[1]))
    plt.savefig(
        plot_directory + '/constraint_function_heatmap.png', format='png', frameon=False, dpi=dpi)
    plt.close('all')
    return 0

    '''




'''
Logs results to .txt in results folder
'''

def log_results(experiment, training_evaluations, violations, best_index, current_directory, descr):
    
    held_out_ll_id = [dat['held_out_ll_indist']
                      for dat in training_evaluations]
    held_out_ll_ood = [dat['held_out_ll_outofdist']
                       for dat in training_evaluations]
    elbos = [dat['elbo'] for dat in training_evaluations]

    output_txt = "\n\nExperiment results : {}  | {}\n".format(
        descr, current_directory)

    output_txt += '\n\n*** Objective function ***\n'
    output_txt += '\nBEST: {}\nAll restarts:\n'.format(
        round(training_evaluations[best_index]['objective'][-1].item(), 4))
    for j, e in enumerate([dat['objective'] for dat in training_evaluations]):
        output_txt += '{}\n'.format(round(e[-1].item(), 4))

    output_txt += '\n\n*** ELBO ***\n'
    output_txt += '\nBEST (by objective): {}\nAll restarts:\n'.format(
        round(elbos[best_index][-1].item(), 4))
    for j, e in enumerate(elbos):
        output_txt += '{}\n'.format(round(e[-1].item(), 4))

    output_txt += '\n\n*** Posterior predictive mass violation ***\n'
    output_txt += '\nBEST (by objective): {}\nAll restarts:\n'.format(
        round(violations[best_index], 4))
    for j, violation in enumerate(violations):
        output_txt += '{}\n'.format(round(violation, 4))

    output_txt += '\n\n*** In-distribution held-out log likelihood ***\n'
    output_txt += '\nBEST (by objective): {}\nAll restarts:\n'.format(
        round(held_out_ll_id[best_index][-1].item(), 4))
    for j, e in enumerate(held_out_ll_id):
        output_txt += '{}\n'.format(round(e[-1].item(), 4))

    output_txt += '\n\n*** Out-of-distribution held-out log likelihood ***\n'
    output_txt += '\nBEST (by objective): {}\nAll restarts:\n'.format(
        round(held_out_ll_ood[best_index][-1].item(), 4))
    for j, e in enumerate(held_out_ll_ood):
        output_txt += '{}\n'.format(round(e[-1].item(), 4))

    # experiment data
    output_txt += '\n\n\n{}\n\n'.format(
        experiment_to_string(experiment))

    text_file = open(current_directory +
                     "/logfile_{}.txt".format(descr), "w")
    text_file.write(output_txt)
    text_file.close()

    return 0
