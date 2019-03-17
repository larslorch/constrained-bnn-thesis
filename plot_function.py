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
size_tup = (3.5, 3.5)  # width, height (inches)

fig, ax = plt.subplots(1, 1, figsize=(size_tup[0], size_tup[1]))
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


x = torch.linspace(-5, 5, steps=1000)
y = 2 * torch.exp(- x.pow(2)) * torch.sin(5 * x)
# y = -0.666667 * x.pow(4) + 4/3  * x.pow(2) + 1
ax.plot(x.numpy(), y.numpy(), '-', color='black', linewidth=1)


ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.set_xlim((-4, 4))
ax.set_ylim((-3, 3))
ax.set_xlabel(r'$x$', fontsize=12)
tmp = ax.set_ylabel(r"$f(x)$", fontsize=12)
tmp.set_rotation(0)
ax.xaxis.set_label_coords(1.05, 0.52)
ax.yaxis.set_label_coords(0.50, 1.05)


plt.tight_layout()
# fig = plt.gcf()  # get current figure

plt.savefig('experiment_results/plot_function.png', format='png', frameon=False, dpi=400)

# plt.show()
