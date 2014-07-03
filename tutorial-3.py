plt.close('all')

from sklearn.cluster import KMeans
from skl_groups.summaries import BagOfWords
bow = BagOfWords(KMeans(n_clusters=6, max_iter=100, n_init=2))
bowized = bow.fit_transform(feats)

from scipy.spatial import Voronoi
from colorized_voronoi import voronoi_finite_polygons_2d

vor = Voronoi(bow.codewords_)
regions, vertices = voronoi_finite_polygons_2d(vor)
vor_cmap = mpl.cm.Pastel1

def plot(X, ax):
    # draw voronoi polygons
    for i, region in enumerate(regions):
        plt.fill(*zip(*vertices[region]),
                 color=vor_cmap(i / float(len(regions))), zorder=0)

    # white out the region outside the "extent of the shape"
    v = gaussian_kde(X.T, .6).evaluate(grid.reshape(2, -1)).reshape(grid.shape[1:])
    plt.contourf(grid[0], grid[1], v, [0, .12, 10],
                 colors=[(1, 1, 1, 1), (0, 0, 0, 0)], zorder=1)

    ax.scatter(*bow.codewords_.T, s=30, color='r')
    ax.scatter(*X.T, s=20, color='k')
    ax.axis('off')
    ax.set_xlim(left, right)
    ax.set_ylim(bot, top)

plt.figure()
shape = (4, n + 1)
for i, X in enumerate(plus):
    ax = plt.subplot2grid(shape, (0, i), rowspan=2)
    plot(X, ax)
    plt.annotate('+', (.5, .15), xycoords='axes fraction',
                 fontsize='large', fontweight='extra bold')

for i, X in enumerate(minus):
    ax = plt.subplot2grid(shape, (2, i), rowspan=2)
    plot(X, ax)
    plt.annotate('-', (.5, .15), xycoords='axes fraction',
                 fontsize='large', fontweight='extra bold')

ax = plt.subplot2grid(shape, (1, n), rowspan=2)
plot(test_pt, ax)
plt.annotate('?', (.5, .2), xycoords='axes fraction',
             fontsize='large', fontweight='extra bold')