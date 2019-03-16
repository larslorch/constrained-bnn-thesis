
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

fig, ax = plt.subplots()
for p in constr_plot:
    ax.add_patch(p.get())
ax.plot(X_plot.squeeze().repeat(y_pred.shape[0], 1).transpose(0, 1).numpy(),
        y_pred.squeeze().transpose(0, 1).numpy(),
        c='blue',
        alpha=0.02)
ax.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
ax.set_title(
    'Function samples for {} BNN using HMC'.format(architecture))
plt.show()

# plt.plot(X_plot.squeeze().numpy(), mean.squeeze().numpy(), c='black')
# plt.fill_between(X_plot.squeeze().numpy(),
#                 (mean - 2 * std).squeeze().numpy(),
#                 (mean + 2 * std).squeeze().numpy(),
#                 color='black',
#                 alpha=0.3)
# plt.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
# plt.title(
#     'Posterior predictive for {} BNN using HMC'.format(architecture))
# plt.show()
