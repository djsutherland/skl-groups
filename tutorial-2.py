plt.close('all')

from skl_groups.summaries import BagMean
means = BagMean().fit_transform(feats)

plt.figure()
for i in range(n):
    plt.annotate('+', means[i], fontsize='large', fontweight='bold', color='blue')
for i in range(n, 2 * n):
    plt.annotate('-', means[i], fontsize='large', fontweight='bold', color='red')
plt.annotate('?', means[-1], fontsize='large', fontweight='bold', color='cyan')
plt.xlim(-.5, .5)
plt.ylim(-.5, .5)
# plt.axis('off')