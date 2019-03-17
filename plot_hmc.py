
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from bnn import make_BNN
from darting_hmc import make_darting_HMC_sampler
from utils import *


title = '6pt_toy_example'

torch.load('experiment_results/hmc_samples/' + title)


'''Approximate posterior predictive for test points'''

y_pred = forward(samples, X_plot)
mean = y_pred.mean(0, keepdim=True)
std = y_pred.std(0, keepdim=True)

'''Approximate posterior predictive for test points'''

fig, ax = plt.subplots()
for p in constr_plot:
    ax.add_patch(p.get())
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

plt.savefig(current_directory + '/hmc/' + experiment['title'] + '_particles.png',
            format='png', frameon=False, dpi=400)
plt.show()
plt.close('all')

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for p in constr_plot:
    ax.add_patch(p.get())
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
plt.savefig(current_directory + '/hmc/' + experiment['title'] + '_filled.png',
            format='png', frameon=False, dpi=400)
plt.show()
