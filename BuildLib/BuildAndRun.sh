#!/usr/bin/env bash
echo curl http://mattmahoney.net/dc/text8.zip > text8.zip
echo unzip text8.zip
echo curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
echo unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
echo rm text8.zip source-archive.zip



TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

python word2vec.py --train_data=text8 --eval_data=questions-words.txt --save_path=/tmp/
  
  