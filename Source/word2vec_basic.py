# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Base import Config
import argparse
import collections
import math
import os

import sys
from tempfile import gettempdir


import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector




from Base import Files
from Source import Base
from Source import DataSet
from Source import Plotter


#data_index = 0
def LogWrite(config: Config.ConfigCls, *log_str_lst):
	log_str :str="";
	for param in log_str_lst:
		log_str = log_str + str(param);

	print(log_str);
	Files.TextWrite(config.OutLogFile(), log_str + '\n');
	return;


def word2vec_basic(config: Config.ConfigCls, identifier:str):
	#global data_index
	out_dir = os.path.join(config.OutDirGet(), identifier)
	config.OutDirSet(out_dir)
	Base.LogDir(out_dir)

	data_index = 0
	LogWrite(config, """\n\n\n\nWord to Vec """ + identifier)
	LogWrite(config, """Example of building, training and visualizing a word2vec model.""")

	###########################################################################
	LogWrite(config,'\n\nStep 1: Download the data.')
	url = config.DownloadUrl();dir=config.DownloadDir(); file = config.DownloadFile(); size = config.DownloadSize();
	filename = Base.maybe_download(dir, url, file, size)

	LogWrite(config,'Read the data into a list of strings.')
	vocabulary = Base.read_data(filename)
	LogWrite(config,'Data size' , len(vocabulary))



	###########################################################################
	LogWrite(config,'\n\nStep 2: Build the dictionary and replace rare words with UNK token.')
	#vocabulary_size = 50000
	vocabulary_size = config.SessionVocSizeGet()
	data, count, unused_dictionary, reverse_dictionary = DataSet.build_dataset(vocabulary, vocabulary_size)
	del vocabulary  # Hint to reduce memory.
	LogWrite(config,'Most common words (+UNK)', count[:5])
	LogWrite(config,'Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])



	###########################################################################
	LogWrite(config,'\n\nStep 3: Function to generate a training batch for the skip-gram model.')
	batch, labels, data_index = DataSet.generate_batch(data=data, data_index=data_index,batch_size=8, num_skips=2, skip_window=1)
	for i in range(8):
		LogWrite(config,batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
		      reverse_dictionary[labels[i, 0]])



	###########################################################################
	LogWrite(config,'\n\nStep 4: Build and train a skip-gram model.')
	batch_size = config.ModelBatchSizeGet()      #128
	embedding_size = config.ModelEmbedSizeGet()  #128  # Dimension of the embedding vector.
	skip_window = config.ModelSkipWindowGet()    #1    # How many words to consider left and right.
	num_skips = config.ModelNumSkipsGet()        #2    # How many times to reuse an input to generate a label.
	num_sampled =config.ModelNumSampledGet()     #64   # Number of negative examples to sample.

	# We pick a random validation set to sample nearest neighbors. Here we limit
	# the validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent. These 3 variables are used only for
	# displaying model accuracy, they don't affect calculation.
	valid_size = config.ValidationSize()      #16   # Random set of words to evaluate similarity on.
	valid_window = config.ValidationWindow()  #100  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)

	graph = tf.Graph()

	with graph.as_default():

		# Input data.
		with tf.name_scope('inputs'):
			train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
			train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
			valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

		# Ops and variables pinned to the CPU because of missing GPU implementation
		with tf.device('/cpu:0'):
			# Look up embeddings for inputs.
			with tf.name_scope('embeddings'):
				embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
				embed = tf.nn.embedding_lookup(embeddings, train_inputs)

			# Construct the variables for the NCE loss
			with tf.name_scope('weights'):
				nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))

			with tf.name_scope('biases'):
				nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		# Explanation of the meaning of NCE loss:
		#   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels,
					inputs=embed, num_sampled=num_sampled,	num_classes=vocabulary_size))

		# Add the loss value as a scalar to summary.
		tf.summary.scalar('loss', loss)

		# Construct the SGD optimizer using a learning rate of 1.0.
		with tf.name_scope('optimizer'):
			optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all
		# embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		# Merge all summaries.
		merged = tf.summary.merge_all()

		# Add variable initializer.
		init = tf.global_variables_initializer()

		# Create a saver.
		saver = tf.train.Saver()



	###########################################################################
	LogWrite(config,'\n\nStep 5: Begin training.')
	#num_steps = 100001

	with tf.Session(graph=graph) as session:
		# Open a writer to write summaries.
		writer = tf.summary.FileWriter(out_dir, session.graph)

		# We must initialize all variables before we use them.
		init.run()
		LogWrite(config,'Initialized')

		average_loss = 0
		num_steps = config.SessionStepsGet();
		for step in xrange(num_steps):
			batch_inputs, batch_labels, data_index = DataSet.generate_batch(data, data_index, batch_size, num_skips, skip_window)
			feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

			# Define metadata variable.
			run_metadata = tf.RunMetadata()

			# We perform one update step by evaluating the optimizer op (including it
			# in the list of returned values for session.run()
			# Also, evaluate the merged op to get all summaries from the returned
			# "summary" variable. Feed metadata variable to session for visualizing
			# the graph in TensorBoard.
			_, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
			average_loss += loss_val

			# Add returned summaries to writer in each step.
			writer.add_summary(summary, step)
			# Add metadata to visualize the graph for the last run.
			if step == (num_steps - 1):
				writer.add_run_metadata(run_metadata, 'step%d' % step)

			loss_step = config.RepLossStep()
			if 0x00 == (step % loss_step):     #2000 == 0:
				if step > 0:
					average_loss /= loss_step  #2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				LogWrite(config,'Average loss at step ', step, ': ', average_loss)
				average_loss = 0

			# Note that this is expensive (~20% slowdown if computed every 500 steps)
			sim_eval_step = config.RepSimStep();
			if 0x00 == (step % sim_eval_step):         #10000 == 0:
				sim = similarity.eval()
				for i in xrange(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = 'Nearest to %s:' % valid_word
					for k in xrange(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = '%s %s,' % (log_str, close_word)
					LogWrite(config,log_str)
		final_embeddings = normalized_embeddings.eval()

		# Write corresponding labels for the embeddings.
		meta_file = config.OutMetaFile();            #  out_dir + '/metadata.tsv'
		with open(meta_file, 'w') as f:
			for i in xrange(vocabulary_size):
				f.write(reverse_dictionary[i] + '\n')

		# Save the model for checkpoints.
		mode_file = config.OutModelFile()            #os.path.join(out_dir, 'model.ckpt')
		saver.save(session, mode_file)

		# Create a configuration for visualizing embeddings with the labels in TensorBoard.
		vis_config = projector.ProjectorConfig()
		embedding_conf = vis_config.embeddings.add()
		embedding_conf.tensor_name = embeddings.name
		embedding_conf.metadata_path = meta_file     #os.path.join(out_dir, 'metadata.tsv')
		projector.visualize_embeddings(writer, vis_config)

	writer.close()






	###########################################################################
	LogWrite(config,'\n\nStep 6: Visualize the embeddings.')
	# pylint: disable=missing-docstring
	# Function to draw visualization of distance between embeddings.
	# def plot_with_labels(low_dim_embs, labels, filename):
	# 	assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
	# 	plt.figure(figsize=(18, 18))  # in inches
	# 	for i, label in enumerate(labels):
	# 		x, y = low_dim_embs[i, :]
	# 		plt.scatter(x, y)
	# 		plt.annotate(
	# 			label,
	# 			xy=(x, y),
	# 			xytext=(5, 2),
	# 			textcoords='offset points',
	# 			ha='right',
	# 			va='bottom')
	#
	# 	plt.savefig(filename)

	# try:
	# 	# pylint: disable=g-import-not-at-top
	# 	from sklearn.manifold import TSNE
	# 	import matplotlib.pyplot as plt
	#
	# 	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
	# 	plot_only = 500
	# 	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
	# 	labels = [reverse_dictionary[i] for i in xrange(plot_only)]
	# 	Plotter.plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
	#
	# except ImportError as ex:
	# 	LogWrite(config,'Please install sklearn, matplotlib, and scipy to show embeddings.')
	# 	LogWrite(config,ex)
	plot_file=config.OutPlotFile()#'tsne.png'
	Plotter.PlotGraph(final_embeddings, reverse_dictionary, plot_file)





print('All functionality is run after tf.compat.v1.app.run() (b/122547914). This')
print('could be split up but the methods are laid sequentially with their usage for clarity.')
def Run(unused_argv):


	# Give a folder path as an argument with '--log_dir' to save
	# TensorBoard summaries. Default is a log folder in current directory.
	current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

	parser = argparse.ArgumentParser()
	parser.add_argument(	'--log_dir',type=str,default=os.path.join(current_path, 'log'),help='The log directory for TensorBoard summaries.')
	flags, unused_flags = parser.parse_known_args()

#	out_dir= os.path.join(config.OutDirGet());
	for i in range(1, 32):
		config: Config.ConfigCls = Config.ConfigCls('./Settings/Config.ini')

		config.ModelEmbedSizeSet(8*i);
		word2vec_basic(config, 'ModelEmbedSize'+str(config.ModelEmbedSizeGet()))



def main(unused_argv):
	Run(unused_argv)
	return;

#def TfRun():
#	tf.app.run()
#	return;

if __name__ == '__main__':
  tf.app.run()
