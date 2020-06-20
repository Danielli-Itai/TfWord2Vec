import os
import sys
import zipfile
from six.moves import urllib
import tensorflow as tf




sys.path.append('../')
sys.path.append('../BaseLib')
from BaseLib import Files




def SaveDir(log_dir:str):
	# Create the directory for TensorBoard variables if there is not.
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

def LogWrite(log_file:str, *log_str_lst):
	log_str :str="";
	for param in log_str_lst:
		log_str = log_str + str(param);

	#print(log_str);
	Files.TextWrite(log_file, log_str + '\n');
	return(log_str);





# pylint: disable=redefined-outer-name
def maybe_download(download_dir:str, url:str, filename:str, expected_bytes:int)->str:
	print("""Download a file if not present, and make sure it's the right size.""")
	local_filename = os.path.join(download_dir, filename)
	if not os.path.exists(local_filename):
		local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
	statinfo = os.stat(local_filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')
	return local_filename

def read_data(filename)->list:
	print("""Extract the first file enclosed in a zip file as a list of words.""")
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data
