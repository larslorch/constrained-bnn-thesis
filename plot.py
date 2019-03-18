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


def plot_posterior_predictive(samples, forward, experiment, current_directory, method='vi'):

    '''Data '''
    X = experiment['data']['X']
    Y = experiment['data']['Y']
    X_plot = experiment['data']['X_plot']
    Y_plot = experiment['data']['Y_plot']
    X_v_id = experiment['data']['X_v_id']
    Y_v_id = experiment['data']['Y_v_id']
    X_v_ood = experiment['data']['X_v_ood']
    Y_v_ood = experiment['data']['Y_v_ood']
    plt_size = experiment['data']['plt_size']

    '''Constraints'''
    plot_patch = experiment['constraints']['plot_patch']
    plot_between = experiment['constraints']['plot_between']

    y_pred = forward(samples, X_plot)
    mean = y_pred.mean(0, keepdim=True)
    std = y_pred.std(0, keepdim=True)


    '''Plotting'''
    fig, ax = plt.subplots(figsize=plt_size)
    for p in plot_patch:
        ax.add_patch(p.get())
    for x, y1, y2 in plot_between:
        ax.fill_between(x.squeeze().numpy(),
                        y1.squeeze().numpy(), y2.squeeze().numpy(), alpha=0.3, color='red')
    ax.plot(X_plot.squeeze().numpy(),
            Y_plot.squeeze().numpy(), c='black', linestyle=':')
    ax.plot(X_plot.squeeze().repeat(y_pred.shape[0], 1).transpose(0, 1).numpy(),
            y_pred.squeeze().transpose(0, 1).numpy(),
            c='blue',
            alpha=0.02,
            linewidth=2)
    ax.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.set_title(
    #         'Function samples for {} BNN using HMC'.format(architecture))
    # plt.savefig(current_directory + '/hmc/' + experiment['title'] + '.png',
    #             format='png', frameon=False, dpi=400)

    ax.set_xlim(experiment['data']['plt_x_domain'])
    ax.set_ylim(experiment['data']['plt_y_domain'])
    # ax.spines['bottom'].set_bounds(-4.5, 4.5)
    bot, top = ax.get_ylim()
    # ax.spines['left'].set_bounds(bot + 0.5, top - 0.5)

    ax.set_xlabel(r"$x$", fontsize=12)
    tmp = ax.set_ylabel(r"$\phi(x; \mathcal{W})$", fontsize=12)
    tmp.set_rotation(0)
    ax.yaxis.set_label_coords(-0.15, 0.50)
    plt.gcf().subplots_adjust(left=0.2)

    plt.savefig(current_directory + '/' + method + '/' + experiment['title'] + '_particles.png',
                format='png', frameon=False, dpi=dpi)
    plt.show()
    plt.close('all')

    #
    fig, ax = plt.subplots(figsize=plt_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for p in plot_patch:
        ax.add_patch(p.get())
    for x, y1, y2 in plot_between:
        ax.fill_between(x.squeeze().numpy(),
                        y1.squeeze().numpy(), y2.squeeze().numpy(), alpha=0.3, color='red')
    ax.plot(X_plot.squeeze().numpy(),
            Y_plot.squeeze().numpy(), c='black', linestyle=':')
    ax.plot(X_plot.squeeze().numpy(), mean.squeeze().numpy(), c='blue')
    ax.fill_between(X_plot.squeeze().numpy(),
                    (mean - 1 * std).squeeze().numpy(),
                    (mean + 1 * std).squeeze().numpy(),
                    color='blue',
                    alpha=0.3)
    ax.fill_between(X_plot.squeeze().numpy(),
                    (mean - 2 * std).squeeze().numpy(),
                    (mean + 2 * std).squeeze().numpy(),
                    color='blue',
                    alpha=0.2)
    ax.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
    ax.set_xlim(experiment['data']['plt_x_domain'])
    ax.set_ylim(experiment['data']['plt_y_domain'])
    # ax.spines['bottom'].set_bounds(-4.5, 4.5)
    bot, top = ax.get_ylim()
    # ax.spines['left'].set_bounds(bot + 0.5, top - 0.5)
    ax.set_xlabel(r"$x$", fontsize=12)

    tmp = ax.set_ylabel(r"$\phi(x; \mathcal{W})$", fontsize=12)
    tmp.set_rotation(0)

    ax.yaxis.set_label_coords(-0.15, 0.50)
    plt.gcf().subplots_adjust(left=0.2)
    # ax.set_title(
    #     'Posterior predictive for {} BNN using HMC'.format(architecture))
    plt.savefig(current_directory + '/' + method + '/' + experiment['title'] + '_filled.png',
                format='png', frameon=False, dpi=dpi)
    plt.show()


'''
Summary plot of training metrics
'''


def plot_training_evaluation(experiment, training_evaluations, current_directory):

    size_tup = experiment['data']['plt_size']

    objs = [dat['objective'] for dat in training_evaluations]
    elbos = [dat['elbo'] for dat in training_evaluations]
    violations = [dat['violation']
                  for dat in training_evaluations]

    cutoff = 1  # ignore first <cutoff> datapoints

    step = experiment['vi']['regular']['reporting_every_']
    start = cutoff * step
    end = (len(objs[0])) * step

    for e in objs:
        plt.plot(
            torch.arange(start=start, end=end, step=step).numpy(), e[cutoff:])
    # plt.title('Objective function ({})'.format(descr))
    # _, top = plt.ylim() # makes bottom end zero
    # plt.ylim((0, top))

    plt.tight_layout()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(size_tup)
    plt.savefig(
        current_directory + '/vi/objective_function.png', format='png', frameon=False, dpi=dpi)
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
    ax[2].set_title('Constraint violation [\%]')
    # TODO these limits only make sense if it is percentage not the violation term from the objective
    ax[2].set_ylim((-0.05, 1.05))

    # plt.suptitle(descr)
    # plt.tight_layout()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches((2 * size_tup[0], size_tup[1]))
    plt.savefig(
        current_directory + '/vi/training.png', format='png', frameon=False, dpi=dpi)
    plt.close('all')


'''
Logs results to .txt in results folder
'''


def log_results(experiment, training_evaluations, violations, best_index, current_directory, descr):

    held_out_ll_id = [dat['held_out_ll_indist']
                      for dat in training_evaluations]
    held_out_ll_ood = [dat['held_out_ll_outofdist']
                       for dat in training_evaluations]

    rmse_id = [dat['rmse_id'] for dat in training_evaluations]
    rmse_ood = [dat['rmse_ood'] for dat in training_evaluations]

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

    output_txt += '\n\n*** In-distribution validation RMSE ***\n'
    output_txt += '\nBEST (by objective): {}\nAll restarts:\n'.format(
        round(rmse_id[best_index][-1].item(), 4))
    for j, e in enumerate(rmse_id):
        output_txt += '{}\n'.format(round(e[-1].item(), 4))

    output_txt += '\n\n*** Out-of-distribution validation RMSE ***\n'
    output_txt += '\nBEST (by objective): {}\nAll restarts:\n'.format(
        round(rmse_ood[best_index][-1].item(), 4))
    for j, e in enumerate(rmse_ood):
        output_txt += '{}\n'.format(round(e[-1].item(), 4))

    # experiment data
    output_txt += '\n\n\n{}\n\n'.format(
        experiment_to_string(experiment))

    text_file = open(current_directory +
                     "/vi/logfile_{}.txt".format(descr), "w")
    text_file.write(output_txt)
    text_file.close()

    return 0
