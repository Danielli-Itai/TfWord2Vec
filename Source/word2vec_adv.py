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

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
from Source import Base
from Source import Options


def LogWrite(options:Options, *log_str_lst):
	return Base.LogWrite(os.path.join(options.save_path, "Log.txt"), log_str_lst);


def forward(options: Options, vocab_size, vocab_counts, examples, labels):
	LogWrite(options, """Build the graph for the forward pass.""")
	# opts = self._options

	# Declare all variables we need.
	# Embedding: [vocab_size, emb_dim]
	init_width = 0.5 / options.emb_dim
	emb = tf.Variable(tf.random.uniform([vocab_size, options.emb_dim], -init_width, init_width), name="emb")

	# Softmax weight: [vocab_size, emb_dim]. Transposed.
	sm_w_t = tf.Variable(tf.zeros([vocab_size, options.emb_dim]), name="sm_w_t")

	# Softmax bias: [vocab_size].
	sm_b = tf.Variable(tf.zeros([vocab_size]), name="sm_b")

	# Global step: scalar, i.e., shape [].
	global_step = tf.Variable(0, name="global_step")

	# Nodes to compute the nce loss w/ candidate sampling.
	labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [options.batch_size, 1])

	# Negative sampling.
	sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(true_classes=labels_matrix,
																				  num_true=1, num_sampled=options.num_samples, unique=True,
																				  range_max=vocab_size,
																				  distortion=0.75, unigrams=vocab_counts.tolist()))

	# Embeddings for examples: [batch_size, emb_dim]
	example_emb = tf.nn.embedding_lookup(emb, examples)

	# Weights for labels: [batch_size, emb_dim]
	true_w = tf.nn.embedding_lookup(sm_w_t, labels)
	# Biases for labels: [batch_size, 1]
	true_b = tf.nn.embedding_lookup(sm_b, labels)

	# Weights for sampled ids: [num_sampled, emb_dim]
	sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
	# Biases for sampled ids: [num_sampled, 1]
	sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

	# True logits: [batch_size, 1]
	true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

	# Sampled logits: [batch_size, num_sampled]
	# We replicate sampled noise labels for all examples in the batch using the matmul.
	sampled_b_vec = tf.reshape(sampled_b, [options.num_samples])
	sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec
	return emb, global_step, true_logits, sampled_logits


def read_analogies(options:Options, word2id:list)-> np.array:
	LogWrite(options, """Reads through the analogy question file.
  	Returns: questions: a [n, 4] numpy array containing the analogy question's word ids.
   questions_skipped: questions skipped due to unknown words.""")
	questions = []
	questions_skipped = 0
	with open(options.eval_data, "rb") as analogy_f:
		for line in analogy_f:
			if line.startswith(b":"):  # Skip comments.
				continue
			words = line.strip().lower().split(b" ")
			ids = [word2id.get(w.strip()) for w in words]
			if None in ids or len(ids) != 4:
				questions_skipped += 1
			else:
				questions.append(np.array(ids))

	LogWrite(options, "Eval analogy file: ", options.eval_data)
	LogWrite(options, "Questions: ", len(questions))
	LogWrite(options, "Skipped: ", questions_skipped)
	analogy_questions = np.array(questions, dtype=np.int32)
	return(analogy_questions);



def save_vocab(options:Options, vocab_size, vocab_words, vocab_counts):
	LogWrite(options, """Save the vocabulary to a file so the model can be reloaded.""")
	#opts = self._options
	with open(os.path.join(options.save_path, "vocab.txt"), "w") as f:
		for i in xrange(vocab_size):
			vocab_word = tf.compat.as_text(vocab_words[i]).encode("utf-8")
			f.write("%s %d\n" % (vocab_word, vocab_counts[i]))
	return;












class Word2Vec(object):
	"""Word2Vec model (Skipgram)."""

	def __init__(self, options:Options, session:tf.compat.v1.Session):
		print("***      Initializing        ***")
		self.my_options:Options = options
		self._session = session
		self.my_word2id = {}
		self.my_id2word = []
	#    self.build_graph(options)
	#    self.build_eval_graph(options)
	#    self.save_vocab(options)





	def LogWrite(self, *log_str_lst):
		Base.LogWrite(os.path.join(self.my_options.save_path, "Log.txt"), log_str_lst);
		return;







	# def read_analogies(self, options:Options):
	#   self.LogWrite("""Reads through the analogy question file.
	#   Returns:
	#     questions: a [n, 4] numpy array containing the analogy question's word ids.
	#     questions_skipped: questions skipped due to unknown words.""")
	#   questions = []
	#   questions_skipped = 0
	#   with open(options.eval_data, "rb") as analogy_f:
	#     for line in analogy_f:
	#       if line.startswith(b":"):  # Skip comments.
	#         continue
	#       words = line.strip().lower().split(b" ")
	#       ids = [self._word2id.get(w.strip()) for w in words]
	#       if None in ids or len(ids) != 4:
	#         questions_skipped += 1
	#       else:
	#         questions.append(np.array(ids))
	#
	#   self.LogWrite("Eval analogy file: ", options.eval_data)
	#   self.LogWrite("Questions: ", len(questions))
	#   self.LogWrite("Skipped: ", questions_skipped)
	#   self._analogy_questions = np.array(questions, dtype=np.int32)



























	def nce_loss(self, options:Options,  true_logits, sampled_logits):
		self.LogWrite("""Build the graph for the NCE loss.""")

		# cross-entropy(logits, labels)
		#opts = self._options
		true_xent = tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.ones_like(true_logits), logits=true_logits)
		sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

		# NCE-loss is the sum of the true and noise (sampled words) contributions, averaged over the batch.
		nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / options.batch_size
		self.my_loss = nce_loss_tensor
		tf.compat.v1.summary.scalar("NCE loss", self.my_loss)
		return nce_loss_tensor



	def forward(self, options:Options, examples, labels):
		self.LogWrite("""Build the graph for the forward pass.""")
		#opts = self._options

		# Declare all variables we need.
		# Embedding: [vocab_size, emb_dim]
		init_width = 0.5 / options.emb_dim
		emb = tf.Variable(tf.random.uniform([self.vocab_size, options.emb_dim], -init_width, init_width), name="emb")
		self._emb = emb

		# Softmax weight: [vocab_size, emb_dim]. Transposed.
		sm_w_t = tf.Variable(tf.zeros([self.vocab_size, options.emb_dim]), name="sm_w_t")

		# Softmax bias: [vocab_size].
		sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

		# Global step: scalar, i.e., shape [].
		self.global_step = tf.Variable(0, name="global_step")

		# Nodes to compute the nce loss w/ candidate sampling.
		labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64),[options.batch_size, 1])

		# Negative sampling.
		sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(true_classes=labels_matrix,
																					  num_true=1, num_sampled=options.num_samples, unique=True, range_max=self.vocab_size,
																					  distortion=0.75, unigrams=self.vocab_counts.tolist()))

		# Embeddings for examples: [batch_size, emb_dim]
		example_emb = tf.nn.embedding_lookup(emb, examples)

		# Weights for labels: [batch_size, emb_dim]
		true_w = tf.nn.embedding_lookup(sm_w_t, labels)
		# Biases for labels: [batch_size, 1]
		true_b = tf.nn.embedding_lookup(sm_b, labels)

		# Weights for sampled ids: [num_sampled, emb_dim]
		sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
		# Biases for sampled ids: [num_sampled, 1]
		sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

		# True logits: [batch_size, 1]
		true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

		# Sampled logits: [batch_size, num_sampled]
		# We replicate sampled noise labels for all examples in the batch using the matmul.
		sampled_b_vec = tf.reshape(sampled_b, [options.num_samples])
		sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec
		return true_logits, sampled_logits




	def build_graph(self, options:Options, word2vec):
		self.LogWrite("""Build the graph for the full model.""")
		#opts = self._options
		# The training data. A text file.
		(words, counts, self.words_per_epoch, self._epoch, self.words, examples, labels) = word2vec.skipgram_word2vec(
			filename=options.train_data, batch_size=options.batch_size, window_size=options.window_size,
			min_count=options.min_count, subsample=options.subsample)
		(self.vocab_words, self.vocab_counts, self.words_per_epoch) = self._session.run([words, counts, self.words_per_epoch])
		self.vocab_size = len(self.vocab_words)
		self.LogWrite("Data file: ", options.train_data)
		self.LogWrite("Vocab size: ", self.vocab_size - 1, " + UNK")
		self.LogWrite("Words per epoch: ", self.words_per_epoch)
		self.my_examples = examples
		self.my_labels = labels
		self.my_id2word = self.vocab_words
		for i, w in enumerate(self.my_id2word):
			self.my_word2id[w] = i

		self.my_emb, self.global_step, true_logits, sampled_logits = forward(options, self.vocab_size, self.vocab_counts, examples, labels)
		return(true_logits, sampled_logits);
		# self.my_loss = self.nce_loss(options, true_logits, sampled_logits)
		# tf.compat.v1.summary.scalar("NCE loss", self.my_loss)
	#     self.optimize(options, self.my_loss)
	#
	#     # Properly initialize all variables.
	# #    tf.global_variables_initializer().run()
	#     self.my_saver = tf.train.Saver()
	#


	def optimize(self, options:Options, loss):
		self.LogWrite("""Build the graph to optimize the loss function.""")

		# Optimizer nodes.
		# Linear learning rate decay.
		#opts = self._options
		words_to_train = float(self.words_per_epoch * options.epochs_to_train)
		self.my_learn_rate = options.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(self.words, tf.float32) / words_to_train)
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.my_learn_rate)
		self.my_train = optimizer.minimize(loss, global_step=self.global_step, gate_gradients=optimizer.GATE_NONE)
		return ;


	def build_eval_graph(self):
		self.LogWrite("""Build the eval graph.""")
		# Eval graph
		# Each analogy task is to predict the 4th word (d) given three words: a, b, c.
		# E.g., a=italy, b=rome, c=france, we should predict d=paris.
		# The eval feeds three vectors of word ids for a, b, c, each of which is of size N,
		# where N is the number of analogies we want to evaluate in one batch.
		analogy_a = tf.compat.v1.placeholder(dtype=tf.int32)  # [N]
		analogy_b = tf.compat.v1.placeholder(dtype=tf.int32)  # [N]
		analogy_c = tf.compat.v1.placeholder(dtype=tf.int32)  # [N]

		# Normalized word embeddings of shape [vocab_size, emb_dim].
		nemb = tf.nn.l2_normalize(self.my_emb, 1)

		# Each row of a_emb, b_emb, c_emb is a word's embedding vector.
		# They all have the shape [N, emb_dim]
		a_emb = tf.gather(nemb, analogy_a)  # a's embs
		b_emb = tf.gather(nemb, analogy_b)  # b's embs
		c_emb = tf.gather(nemb, analogy_c)  # c's embs

		# We expect that d's embedding vectors on the unit hyper-sphere is
		# near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
		target = c_emb + (b_emb - a_emb)

		# Compute cosine distance between each pair of target and vocab dist has shape [N, vocab_size].
		dist = tf.matmul(target, nemb, transpose_b=True)

		# For each question (row in dist), find the top 4 words.
		_, pred_idx = tf.nn.top_k(dist, 4)

		# Nodes for computing neighbors for a given word according to their cosine distance.
		nearby_word = tf.compat.v1.placeholder(dtype=tf.int32)  # word id
		nearby_emb = tf.gather(nemb, nearby_word)
		nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
		nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, self.vocab_size))

		# Nodes in the construct graph which are used by training and evaluation to run/feed/fetch.
		self._analogy_a = analogy_a
		self._analogy_b = analogy_b
		self._analogy_c = analogy_c
		self._analogy_pred_idx = pred_idx
		self._nearby_word = nearby_word
		self._nearby_val = nearby_val
		self._nearby_idx = nearby_idx
		return;



	# def save_vocab(self, options:Options, vocab_size, vocab_words, vocab_counts):
	#   LogWrite(options, """Save the vocabulary to a file so the model can be reloaded.""")
	#   #opts = self._options
	#   with open(os.path.join(options.save_path, "vocab.txt"), "w") as f:
	#     for i in xrange(vocab_size):
	#       vocab_word = tf.compat.as_text(vocab_words[i]).encode("utf-8")
	#       f.write("%s %d\n" % (vocab_word, vocab_counts[i]))
	#   return;




	def _train_thread(self):
		initial_epoch, = self._session.run([self._epoch])
		while True:
			_, epoch = self._session.run([self.my_train, self._epoch])
			if epoch != initial_epoch:
				break


	def train(self, options:Options):
		self.LogWrite("""Train the model.""")
		#opts = self._options

		initial_epoch, initial_words = self._session.run([self._epoch, self.words])

		summary_op = tf.compat.v1.summary.merge_all()
		summary_writer = tf.compat.v1.summary.FileWriter(options.save_path, self._session.graph)
		workers = []
		for _ in xrange(options.concurrent_steps):
			t = threading.Thread(target=self._train_thread)
			t.start()
			workers.append(t)

		last_words, last_time, last_summary_time = initial_words, time.time(), 0
		last_checkpoint_time = 0
		while True:
			time.sleep(options.statistics_interval)  # Reports our progress once a while.
			(epoch, step, loss, words, lr) = self._session.run([self._epoch, self.global_step, self.my_loss, self.words, self.my_learn_rate])
			now = time.time()
			last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
			self.LogWrite("Epoch %4d, Step %8d:\t learn-rate=%5.3f,\t loss = %6.2f,\t words/sec = %8.0f\r" % (epoch, step, lr, loss, rate))
			sys.stdout.flush()
			if now - last_summary_time > options.summary_interval:
				summary_str = self._session.run(summary_op)
				summary_writer.add_summary(summary_str, step)
				last_summary_time = now
			if now - last_checkpoint_time > options.checkpoint_interval:
				self.my_saver.save(self._session, os.path.join(options.save_path, "model.ckpt"), global_step=step.astype(int))
				last_checkpoint_time = now
			if epoch != initial_epoch:
				break

		for t in workers:
			t.join()

		return epoch











	def analogy_calc(self, analogy):
		self.LogWrite("""Predict the top 4 answers for analogy questions.""")
		idx, = self._session.run([self._analogy_pred_idx], {
			self._analogy_a: analogy[:, 0], self._analogy_b: analogy[:, 1], self._analogy_c: analogy[:, 2]} )
		return idx

	def analogys_evaluate(self, analogy_questions):
		self.LogWrite("""Evaluate analogy questions and reports accuracy.""")

		# How many questions we get right at precision@1.
		correct = 0

		try:
			total = analogy_questions.shape[0]
		except AttributeError as e:
			raise AttributeError("Need to read analogy questions.")

		start = 0
		while start < total:
			limit = start + 2500
			sub = analogy_questions[start:limit, :]
			idx = self.analogy_calc(sub)
			start = limit
			for question in xrange(sub.shape[0]):
				for j in xrange(4):
					if idx[question, j] == sub[question, 3]:
						# Bingo! We predict_cls correctly. E.g., [italy, rome, france, paris].
						correct += 1
						break
					elif idx[question, j] in sub[question, :3]:
						# We need to skip words already in the question.
						continue
					else:
						# The correct label is not the precision@1
						break
		self.LogWrite()
		self.LogWrite("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))





	def analogy_predict(self, w0, w1, w2):
		self.LogWrite("""Predict word w3 as in w0:w1 vs w2:w3.""")
		wid = np.array([[self.my_word2id.get(w, 0) for w in [w0, w1, w2]]])
		idx = self.analogy_calc(wid)
		for c in [self.my_id2word[i] for i in idx[0, :]]:
			if c not in [w0, w1, w2]:
				self.LogWrite(c)
				return
		self.LogWrite("unknown")





	def nearby_calc(self, words, num=20):
		self.LogWrite("""self.LogWrites out nearby words given a list of words.""")
		ids = np.array([self.my_word2id.get(x, 0) for x in words])
		vals, idx = self._session.run([self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
		for i in xrange(len(words)):
			self.LogWrite("\n%s\n=====================================" % (words[i]))
			for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
				self.LogWrite("%-20s %6.4f" % (self.my_id2word[neighbor], distance))





def _start_shell(local_ns=None):
	# An interactive shell is useful for debugging/development.
	import IPython
	user_ns = {}
	if local_ns:
		user_ns.update(local_ns)
	user_ns.update(globals())
	IPython.start_ipython(argv=[], user_ns=user_ns)





def Run(opts:Options.Options):
	"""Train a word2vec model."""
	#  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
	#    print("--train_data --eval_data and --save_path must be specified.")
	#    sys.exit(1)
	path:str = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../BuildLib/word2vec_ops.so')
	word2vec = tf.load_op_library(path)

	#  config: Config.ConfigCls = Config.ConfigCls('./Settings/ConfigAdv.ini')
	with tf.Graph().as_default(), tf.compat.v1.Session() as session:
		#with tf.device("/cpu:0"):
		with tf.device("/cpu:0"):
			model = Word2Vec(opts,session)
			(true_logits, sampled_logits) = model.build_graph(opts, word2vec)
			model.nce_loss(opts, true_logits, sampled_logits)
			model.optimize(opts, model.my_loss)

			# Properly initialize all variables.
			#    tf.global_variables_initializer().run()
			model.my_saver = tf.compat.v1.train.Saver()
			tf.compat.v1.global_variables_initializer().run()   # Properly initialize all variables.

			model.build_eval_graph()
			save_vocab(opts, model.vocab_size, model.vocab_words, model.vocab_counts)

			analogy_questions = read_analogies(opts, model.my_word2id)  # Read analogy questions
			for epoc_num in xrange(opts.epochs_to_train):
				LogWrite(opts, "*** Epoc run: {0} of {1} train and evaluate ***".format(epoc_num, opts.epochs_to_train));
				model.train(opts)  # Process one epoch
				model.analogys_evaluate(analogy_questions)  # Eval analogies.

			LogWrite(opts, "*** Perform a final save.");
			model.my_saver.save(session, os.path.join(opts.save_path, "model.ckpt"), global_step=model.global_step)
			if Options.FLAGS.interactive:
				# E.g.,
				model.analogy_predict(b'france', b'paris', b'russia')
				model.nearby_calc([b'proton', b'elephant', b'maxwell'])
				_start_shell(locals())


#def main(_):
def main(unused_argv):

	opts = Options.Options()
	current_path = opts.save_path


	#for i in range(19, 32):
	for i in range (1,8):
		np.random.seed(seed=30)
		opts.emb_dim = 8*i
		opts.window_size = 5
		opts.save_path = os.path.join(current_path, 'ModelEmbedSize' + str(opts.emb_dim))
		Base.SaveDir(opts.save_path)
		Run(opts)


if __name__ == "__main__":
	tf.compat.v1.app.run()
