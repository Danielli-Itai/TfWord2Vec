##############################################################
# Files.spy
#
# Curl interface module
#
# License:  MIT 2019 Itai Danielli
##############################################################
import os
import shutil

from tempfile import gettempdir
from pathlib import Path
import configparser



def TempDir():
   return(gettempdir());

# Returns current working directory of a process.
def WorkingDir()->str:
   return(os.getcwd() + os.path.sep)






#  Text File handlers API
def TextRead(file_name:str)->str:
   file_path = WorkingDir() + file_name;
   f = open(file_path, 'r')
   return(f.read())


#Splitting text in file name text by using new line
def TextReadLines(file_name:str)->str:
   return(TextRead(file_name).split('\n'));


#Configuration of writting file text
def TextWrite(file_name:str,txt:str)->bool:
   result :bool = True
   try:
      f = open(file_name,"a")
      f.write(txt);
      f.close();
   except:
      result = False
   return(result)








#Configuration of config.ini file process
def IniRed(file_name:str)->list:
   result :bool = True
   try:
      config = configparser.ConfigParser()
      config.read('config.ini')
   except:
      result = False
   return(result)





from multiprocessing.pool import Pool
def path_traveler(path):
   for root, _, files in os.walk(path):
      for name in files:
         yield os.path.join(root, name)


def dir_traveler(path):
   for entry in os.scandir(path):
      if entry.is_file():
         yield entry.path


def load_file(path):
   with open(path, 'r', encoding='utf-8') as f:
      return f.read(), f.name


def load_parallel(paths, num_of_workers=8, chunk_size=8):
   with Pool(num_of_workers) as pool:
      for f in pool.imap(load_file, paths, chunksize=chunk_size):
         yield f[0]


def RemoveDir(path:str)->bool:
   result:bool = True
   try: shutil.rmtree(path);
   except: result = False;  pass;
   return(result);


def DirExportText(source_path:str, export_path:str, extentions)->list:
   for root, dirs, files in os.walk(source_path, topdown=False):
      files_copied:int = 000;
      curr_dir = os.path.relpath(root,source_path)
      RemoveDir(os.path.join(export_path, curr_dir));
      os.makedirs(os.path.join(export_path, curr_dir));
      for name in files:
         print(os.path.splitext(name)[1])
         if(os.path.splitext(name)[1]) in extentions:
            print(os.path.join(root, name))
            dest_path = os.path.join(export_path, curr_dir);
            shutil.copy(os.path.join(root, name),os.path.join(dest_path, name))
            files_copied = files_copied + 1
      if(0x00 == files_copied):
         RemoveDir(os.path.join(export_path, curr_dir));

      for name in dirs:
         DirExportText(os.path.join(root, name))
   # for filename in os.listdir(source_path):
   #    os.path.isfile(source_path + filename)
   #    if filename.endswith(".txt"):
   #       f = open(filename)
   #       lines = f.read()
   #       print(lines[10])
   #       continue
   #    else:
   #       continue
   return;