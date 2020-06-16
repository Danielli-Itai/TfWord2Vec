import sys
import tensorflow as tf
from multiprocessing import Process





#Import project libraries.
sys.path.append('/BaseLib')
sys.path.append('/BuildLib')
sys.path.append('/Settings')
sys.path.append('/Source')
sys.path.append('/SummeryAdv')
sys.path.append('/SummeryBasic')
from Source import word2vec_basic




#Main entry function.
process:Process = None;
def main(unused_argv):
#	word2vec_basic.Run(unused_argv)
	process = Process(target=word2vec_basic.Run, args=(unused_argv,))
	process.start()
	process.join()
	return;





if __name__ == '__main__':
	tf.app.run()