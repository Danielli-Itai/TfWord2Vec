import tensorflow as tf
from Source import word2vec_basic





def main(unused_argv):
	word2vec_basic.Run(unused_argv)
	return;

if __name__ == '__main__':
	tf.app.run()