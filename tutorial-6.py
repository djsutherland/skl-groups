plt.close('all')

from skl_groups import kernels
K = kernels.Symmetrize().fit_transform(divs[0, 0])
K = kernels.RBFize(gamma=1, scale_by_median=True).fit_transform(K)
K = kernels.ProjectPSD().fit_transform(K)
hidim = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2)
lowdim = lle.fit_transform(hidim)

for i in range(n):
    plt.annotate('+', lowdim[i], fontsize='large', fontweight='bold', color='blue')
for i in range(n, 2 * n):
    plt.annotate('-', lowdim[i], fontsize='large', fontweight='bold', color='red')
plt.annotate('?', lowdim[-1], fontsize='large', fontweight='bold', color='cyan')

plt.xlim(lowdim[:, 0].min() - .1, lowdim[:, 0].max() + .1)
plt.ylim(lowdim[:, 1].min() - .1, lowdim[:, 1].max() + .1)
plt.axis('off')