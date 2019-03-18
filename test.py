import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import torch


# def label(xy, text):
#     y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
#     plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


fig, ax = plt.subplots()
# create 3x3 grid to plot the artists
# grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T

# patches = []


# # add a path patch
# Path = mpath.Path
# path_data = [
#     (Path.MOVETO, [0.018, -0.11]),
#     (Path.CURVE4, [-0.031, -0.051]),
#     (Path.CURVE4, [-0.115, 0.073]),
#     (Path.CURVE4, [-0.03, 0.073]),
#     (Path.LINETO, [-0.011, 0.039]),
#     (Path.CURVE4, [0.043, 0.121]),
#     (Path.CURVE4, [0.075, -0.005]),
#     (Path.CURVE4, [0.035, -0.027]),
#     (Path.CLOSEPOLY, [0.018, -0.11])]
# codes, verts = zip(*path_data)
# path = mpath.Path(verts, codes)
# patch = mpatches.PathPatch(path)
# patches.append(patch)


# colors = np.linspace(0, 1, len(patches))
# collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
# collection.set_array(np.array(colors))
# ax.add_collection(collection)
# # ax.add_line(line)

# plt.axis('equal')
# plt.axis('off')
# plt.tight_layout()


def make_fill_object(x, y1, y2):
    fig, ax = plt.subplots()
    polycoll = ax.fill_between(x.numpy(), y1.numpy(), y2.numpy())
    plt.close('all')
    return polycoll

x = torch.linspace(-5, 5, steps = 100)
y1 = torch.sin(x)
y2 = 2 * torch.ones(x.shape)

# ax.plot(x.numpy(), y.numpy())

# polycoll = ax.fill_between(x.numpy(), y1.numpy(), y2.numpy())
# print(polycoll)

fig, ax = plt.subplots()
polycoll = make_fill_object(x, y1, y2)
ax.add_collection(polycoll)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.show()
