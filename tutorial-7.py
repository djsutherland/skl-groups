plt.close('all')

from skl_groups.preprocessing import BagMinMaxScaler
from skl_groups.summaries import L2DensityTransformer

scaled = BagMinMaxScaler((0, 1), truncate=True).fit_transform(feats)
l2ized = L2DensityTransformer(smoothness=5).fit_transform(scaled)

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2)
lowdim = lle.fit_transform(l2ized)

for i in range(n):
    plt.annotate('+', lowdim[i], fontsize='large', fontweight='bold', color='blue')
for i in range(n, 2 * n):
    plt.annotate('-', lowdim[i], fontsize='large', fontweight='bold', color='red')
plt.annotate('?', lowdim[-1], fontsize='large', fontweight='bold', color='cyan')

plt.xlim(lowdim[:, 0].min() - .1, lowdim[:, 0].max() + .1)
plt.ylim(lowdim[:, 1].min() - .1, lowdim[:, 1].max() + .1)
plt.axis('off')