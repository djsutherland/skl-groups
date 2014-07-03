plt.close('all')

from skl_groups.divergences import KNNDivergenceEstimator
knn_div = KNNDivergenceEstimator(div_funcs=['kl'], Ks=[2])
divs = knn_div.fit_transform(feats)

plt.matshow(divs[0, 0], cmap='hot')
plt.colorbar()