import os
import tensorflow as tf






flags = tf.app.flags
flags.DEFINE_string("save_path", "./SummeryAdv", "Directory to write the model and training summaries.")
flags.DEFINE_string("train_data", "./_Data/text8.txt", "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("eval_data", "./_Data/questions-words.txt", "File consisting of analogies of four tokens."
                                                                "embedding 2 - embedding 1 + embedding 3 should be close to embedding 4."
                                                                "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 5,"Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100, "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 128, "Number of training examples processed per step (size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12, "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 1, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean("interactive", False, "If true, enters an IPython interactive session to play with the trained "
                                          "model. E.g., try model.analogy(b'france', b'paris', b'russia') and model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5, "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5, "Save training summary to file every n seconds (rounded up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600, "Checkpoint the model (i.e. save the parameters) every n seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS





class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options - The training text file.
    self.train_data = FLAGS.train_data

    # Training options - Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # Training options - The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Training options - Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Training options - Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Training options - Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # Training options - The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # Training options - The minimum number of word occurrences for it to be included in the vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest statistics_interval).
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints (rounds up to the nearest statistics interval).
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    # Eval options.
    # The text file for eval.
    self.eval_data = FLAGS.eval_data
