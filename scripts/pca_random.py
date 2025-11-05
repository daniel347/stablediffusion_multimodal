import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

latents = np.random.randn(12, 768)

pca = PCA(n_components=2).fit(latents)
latents_proj = pca.transform(latents)
print("===== PCA =====")
print(f"explained variance ratio: {pca.explained_variance_ratio_}")

tsne = TSNE(n_components=2, perplexity=10).fit(latents)
latents_tsne = tsne.fit_transform(latents)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("PCA")
sns.scatterplot(x=latents_proj[:, 0], y=latents_proj[:, 1], ax=ax[0])
ax[1].set_title("TSNE")
sns.scatterplot(x=latents_tsne[:, 0], y=latents_tsne[:, 1], ax=ax[1])

fig.tight_layout()
fig.show()

# Also compute the spectrum of PCA component explained variance ratios
pca_full = PCA().fit(latents)
explained_var = pca_full.explained_variance_ratio_
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.semilogy(explained_var)
ax.set_title("PCA explained variance ratios")
fig.show()