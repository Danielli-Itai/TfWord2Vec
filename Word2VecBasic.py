import sys
from multiprocessing import Process
import tensorflow.compat.v1 as tf




#Import project libraries.
sys.path.append('/BaseLib')
sys.path.append('/BuildLib')
sys.path.append('/Settings')
sys.path.append('/Source')
sys.path.append('/SummeryAdv')
sys.path.append('/SummeryBasic')
from Source import word2vec_basic


from multiprocessing.connection import Client
from array import array
def ClientConnect(pipe_name:str, pipe_id:int, key:str):
	connection:Client = None;
	connected:bool = True

	address = (pipe_name, pipe_id)
	try:
		connection = Client(address, authkey=key)
		#conn.close()
	except:
		connected = False
	return(connection, connected)

#print(connection.recv())  # => [2.25, None, 'junk', float]
def ClientReceive(connection):
	return(connection.rcv())

#print(connection.recv_bytes())  # => 'hello'
#arr = array('i', [0, 0, 0, 0, 0])
#print(connection.recv_bytes_into(arr))  # => 8
#print(arr)  # => array('i', [42, 1729, 0, 0, 0])
def ClientRecvdBytes(connection, msg:bytes)->bool:
	size:int = 0
	try:
		size = connection.recv_bytes_into(msg)
	except:
		print('Connection lost')
	return(size)

def ClientDisconnect(connection):
	connection.close()





#Main entry function.
process:Process = None;
def main(unused_argv):
#	word2vec_basic.Run(unused_argv)
	process = Process(target=word2vec_basic.Run, args=(unused_argv,))
	process.start()

	connection = None
	connected = False
	while not connected:
		(connection, connected) = ClientConnect(word2vec_basic.PIPE_NAME, word2vec_basic.PIPE_ID, word2vec_basic.PIPE_KEY)


	msg: str = ""
	while(word2vec_basic.MSG_EXIT != msg):
		rcv_buff:bytes = bytearray(word2vec_basic.MSG_SIZE)
		size:int = ClientRecvdBytes(connection, rcv_buff);
		if(0<size):
			msg = rcv_buff.decode()
			print(msg)
		else:
			break

	process.join()
	return;




if __name__ == '__main__':
#	argsv = sys.argv[1:]
#	main(argsv)
	tf.app.run()