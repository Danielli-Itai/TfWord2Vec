from six.moves import xrange  # pylint: disable=redefined-builtin

import os
from tempfile import gettempdir
import matplotlib.pyplot as plt
# pylint: disable=g-import-not-at-top
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_with_labels(low_dim_embs, labels, filename):
	assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
	plt.figure(figsize=(18, 18))  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig(filename)



def PlotGraph(final_embeddings, reverse_dictionary, plot_file):
	try:
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
		plot_only = 500
		low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
		labels = [reverse_dictionary[i] for i in xrange(plot_only)]

		plot_with_labels(low_dim_embs, labels, plot_file)

	except ImportError as ex:
		print('Please install sklearn, matplotlib, and scipy to show embeddings.')
		print(ex)
	return;