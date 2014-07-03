import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# generate data
np.random.seed(2)
n = 3

mvn = np.random.multivariate_normal
def nudge(k):
    nudge = np.zeros((k, 2))
    nudge[:k//2, 0] = -.5
    nudge[k//2:, 0] = .5
    return nudge
plus = [mvn([0, 0], [[.1**2,0],[0,.35**2]], size=k) + nudge(k)
        for k in np.random.randint(8, 20, size=n)]
minus = [mvn([0, 0], [[.2**2,0],[0,.2**2]], size=k)
        for k in np.random.randint(8, 20, size=n)]
test_pt = mvn([0, 0], [[.2**2,0],[0,.2**2]], size=np.random.randint(5, 15))


# plot
from scipy.stats import gaussian_kde

left = -1.1
right = 1.1
top = 1.3
bot = -1.3
grid = np.mgrid[left:right:.01, bot:top+.01:.01]

def plot(X, ax):
    v = gaussian_kde(X.T, .6).evaluate(grid.reshape(2, -1)).reshape(grid.shape[1:])
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0
    ax.contour(grid[0], grid[1], v, [.12], cmap='hot')
    ax.scatter(*X.T, s=50, color='k')
    ax.axis('off')
    ax.set_xlim(left, right)
    ax.set_ylim(bot, top)

plt.figure()
shape = (4, n + 1)
for i, X in enumerate(plus):
    ax = plt.subplot2grid(shape, (0, i), rowspan=2)
    # plt.subplot(2, n, i)
    plot(X, ax)
    plt.annotate('+', (.5, .15), xycoords='axes fraction',
                 fontsize='large', fontweight='extra bold')

for i, X in enumerate(minus):
    ax = plt.subplot2grid(shape, (2, i), rowspan=2)
    # plt.subplot(2, n, n + i)
    plot(X, ax)
    plt.annotate('-', (.5, .15), xycoords='axes fraction',
                 fontsize='large', fontweight='extra bold')

ax = plt.subplot2grid(shape, (1, n), rowspan=2)
plot(test_pt, ax)
plt.annotate('?', (.5, .2), xycoords='axes fraction',
             fontsize='large', fontweight='extra bold')


# for later
bags = plus + minus + [test_pt]
labels = np.r_[np.ones(n, dtype=bool), np.zeros(n, dtype=bool), False]

from skl_groups.features import Features
feats = Features(bags, labels=labels)