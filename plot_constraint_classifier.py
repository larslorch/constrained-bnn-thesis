import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import math
from pprint import pprint
# from matplotlib import rc
import torch
from utils import *
from plot import *

# tau_s < tau_b
size_tup = (2.2, 2.2)  # width, height (inches)
taus = [(1.0, 3.0), (1.0, 10.0), (0.5, 3.0), (0.5, 10.0)]

fig, ax = plt.subplots(1, 4, sharey=True, figsize=(4 * size_tup[0], size_tup[1]))
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

for j, tup in enumerate(taus):
    tau_s, tau_b = tup

    x = torch.linspace(-5, 5, steps=200)
    y = psi(x, tau_s, tau_b)
    ax[j].plot(x.numpy(), y.numpy(), 'black')
    ax[j].plot(x.numpy(), 0.5 * torch.tanh(-x).numpy() + 0.5, '--', color='blue', linewidth=1)


    ax[j].spines['top'].set_color('none')
    ax[j].spines['right'].set_color('none')
    ax[j].spines['left'].set_position(('data', -3))
    ax[j].set_xlim((-3, 3))
    ax[j].set_ylim((-0.2, 1.2))
    ax[j].set_xlabel(r'$z$', fontsize=12)
    if j == 0:        
        tmp = ax[j].set_ylabel(r"$\Psi_{\tau_b, \tau_s}(z)$", fontsize=12)
        # tmp.set_rotation(0)
    else:

        # ax[j].spines['left'].set_color('none')

        pass
    # ax[j].set_ylabel('')


plt.tight_layout()
# fig = plt.gcf()  # get current figure

plt.savefig('experiment_results/plot_constraint_classifier.png', format='png', frameon=False, dpi=400)

# plt.show()
