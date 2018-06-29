"""
Tutorial for Tensorflow RNN Models
=========================
This article is an example for deploying tensorflow RNN models with NNVM.
"""
import nnvm
import tvm
import numpy as np
import os

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
from tensorflow.python.framework import graph_util
dir(tf.contrib)


sample_repo = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
sample_data_file = 'simple-examples.tgz'
sample_url = sample_repo+sample_data_file

ptb_repo = 'https://github.com/joyalbin/dmlc_store/raw/master/trained-models/tf/ptb/pb/'
ptb_model_file = 'ptb_model_with_shapes.pb'
ptb_model_url = ptb_repo+ptb_model_file


###############################################################################
# Download PTB model files.and input sample data
# ---------------------------------------------
from tvm.contrib.download import download

DATA_DIR = './ptb_pb/'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

download(sample_url, DATA_DIR+sample_data_file)
download(ptb_model_url, DATA_DIR+ptb_model_file)

import tarfile
t = tarfile.open(DATA_DIR+sample_data_file, 'r')
t.extractall(DATA_DIR)


###############################################################################
# Read the PTB sample data input to create vocabulary
# ---------------------------------------------
import collections
import os

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().encode("utf-8").decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    #for python 3.x
    id_to_word = dict((v, k) for k, v in word_to_id.items())
    return word_to_id, id_to_word

def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    word_to_id, id_2_word = _build_vocab(train_path)
    return word_to_id, id_2_word


###############################################################################
# Vocabulary
# ---------------------------------------------
# Create vocabulary from the input sample data provided
raw_data = ptb_raw_data(DATA_DIR+'simple-examples/data/')
word_to_id, id_to_word = raw_data
vocab_size = len(word_to_id)


###############################################################################
# PTB Configuration
# ---------------------------------------------
# PTB test model configurations for sampling. This config should match with
#the config used to train the model
class SmallConfig(object):
    """Small config."""
    num_layers = 2
    num_steps = 1
    hidden_size = 200
    batch_size = 1
    vocab_size = 10000

def get_config():
    return SmallConfig()


###############################################################################
# Restore Model
# ---------------------------------------------
# Restore the PTB pre-trained model checkpoints
import nnvm.testing.tf
input_binary = True
ptb_model_file = DATA_DIR+'ptb_model_with_shapes.pb'
mode = "rb" if input_binary else "r"
with tf.gfile.FastGFile(ptb_model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    final_graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

sym, params = nnvm.frontend.from_tensorflow(final_graph_def)


######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.
import nnvm.compiler
target = 'llvm'
config = get_config()
batch_size = config.batch_size
num_steps = config.num_steps
num_hidden = config.hidden_size
num_layers = config.num_layers
input_shape = (batch_size, num_steps)
output_shape = (batch_size, num_hidden)
shape_dict = {'Model/Placeholder': input_shape,
              'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c':(num_layers, batch_size, num_hidden),
              'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h':(num_layers, batch_size, num_hidden)}
dtype_dict = {'Model/Placeholder': 'int32',
              'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c':'float32',
              'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h':'float32'}

graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                         dtype=dtype_dict, params=params)


######################################################################
# Execute on TVM
# ---------------------------------------------
# The process is no different from other example
from tvm.contrib import graph_runtime

ctx = tvm.cpu(0)
out_dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)


######################################################################
# Predition
# ---------------------------------------------
# Create sample rediction results
batch_size = config.batch_size
vocab_size = config.vocab_size
def pick_from_weight(weight, pows=1.0):
    weight = weight**pows
    t = np.cumsum(weight)
    s = np.sum(weight)
    #return int(np.searchsorted(t, np.random.rand(1) * s))
    return int(np.searchsorted(t, 0.5 * s))

out_sample_shape = (batch_size, vocab_size)
out_state_shape = (num_layers, 2, batch_size, num_hidden)

def do_sample(model, data, in_states, num_samples):
    """Sampled from the model"""
    samples = []
    state = in_states
    sample = None
    for x in data:
        word_id = np.full((batch_size, num_steps), x, dtype="int32")
        model.set_input('Model/Placeholder', tvm.nd.array(word_id.astype("int32")))
        in_state_tup = np.split(state, indices_or_sections=2, axis=1)
        in_state_c = np.reshape(in_state_tup[0], (num_layers, batch_size, num_hidden))
        in_state_h = np.reshape(in_state_tup[1], (num_layers, batch_size, num_hidden))
        model.set_input('Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c',
                        tvm.nd.array(in_state_c.astype("float32")))
        model.set_input('Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h',
                        tvm.nd.array(in_state_h.astype("float32")))
        model.set_input(**params)
        model.run()
        tvm_output = model.get_output(0, tvm.nd.empty(out_sample_shape,
                                                      out_dtype)).asnumpy()
        state_output = model.get_output(1, tvm.nd.empty(out_state_shape,
                                                        out_dtype)).asnumpy()
        state = state_output
        sample = pick_from_weight(tvm_output[0])

    if sample is not None:
        samples.append(sample)
    else:
        samples.append(0)

    k = 1
    while k < num_samples:
        word_id = np.full((batch_size, num_steps), samples[-1], dtype="int32")
        model.set_input('Model/Placeholder', tvm.nd.array(word_id.astype("int32")))
        in_state_tup = np.split(state, indices_or_sections=2, axis=1)
        in_state_c = np.reshape(in_state_tup[0], (num_layers, batch_size, num_hidden))
        in_state_h = np.reshape(in_state_tup[1], (num_layers, batch_size, num_hidden))
        model.set_input('Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c',
                        tvm.nd.array(in_state_c.astype("float32")))
        model.set_input('Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h',
                        tvm.nd.array(in_state_h.astype("float32")))
        model.set_input(**params)
        model.run()
        tvm_output = model.get_output(0, tvm.nd.empty(out_sample_shape,
                                                      out_dtype)).asnumpy()
        state_output = model.get_output(1, tvm.nd.empty(out_state_shape,
                                                        out_dtype)).asnumpy()
        state = state_output
        sample = pick_from_weight(tvm_output[0])
        samples.append(sample)
        k += 1

    return samples, state

def pretty_print(items, is_char_model, id2word):
    if not is_char_model:
        return ' '.join([id2word[x] for x in items])
    else:
        return ''.join([id2word[x] for x in items]).replace('_', ' ')


###############################################################################
# Input words
# ---------------------------------------------
# The input data provide the context to predict next word
from sys import version_info
while True:
    if version_info[0] < 3:
        inpt = raw_input("Enter your sample prefix: ")
        cnt = int(raw_input("Sample size: "))
    else:
        #python 3.x
        inpt = input("Enter your sample prefix: ")
        cnt = int(input("Sample size: "))

    in_state = np.full((num_layers, 2, batch_size, num_hidden), 0, dtype="float32")
    seed_for_sample = inpt.split()
    print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample],
                                    False, id_to_word))
    samples, _ = do_sample(m, [word_to_id[word] for word in seed_for_sample],
                           in_state, cnt)
    print("Sample: %s" % pretty_print(samples, False, id_to_word))
