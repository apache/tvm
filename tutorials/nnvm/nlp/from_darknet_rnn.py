"""
Compile Darknet Models for RNN
==============================
This article is an introductory tutorial to deploy darknet rnn models with NNVM.

This script will run a character prediction model
Each module consists of 3 fully-connected layers. The input layer propagates information from the
input to the current state. The recurrent layer propagates information through time from the
previous state to the current one.

The input to the network is a 1-hot encoding of ASCII characters. We train the network to predict
the next character in a stream of characters. The output is constrained to be a probability
distribution using a softmax layer.

Since each recurrent layer contains information about the current character and the past
characters, it can use this context to predict the future characters in a word or phrase.

All the required models and libraries will be downloaded from the internet
by the script.
"""
import random
import os
import sys
import time
import urllib
import requests
import numpy as np
import urllib.request as urllib2
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.darknet import __darknetffi__
import nnvm
import nnvm.frontend.darknet

MODEL_NAME = 'rnn' #Model name
seed = 'Thus' #Seed value
num = 1000 #Number of characters to predict

# Download cfg and weights file if first time.
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
WEIGHTS_URL = REPO_URL + 'weights/' + WEIGHTS_NAME + '?raw=true'

def _dl_progress(count, block_size, total_size):
    """Show the download progress."""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def _download(url, path, overwrite=False, sizecompare=False):
    """Downloads the file from the internet.
    """
    if os.path.isfile(path) and not overwrite:
        if sizecompare:
            file_size = os.path.getsize(path)
            res_head = requests.head(url)
            res_get = requests.get(url, stream=True)
            if 'Content-Length' not in res_head.headers:
                res_get = urllib2.urlopen(url)
            url_file_size = int(res_get.headers['Content-Length'])
            if url_file_size != file_size:
                print("exist file got corrupted, downloading", path, " file freshly")
                _download(url, path, True, False)
                return
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path, reporthook=_dl_progress)
        print('')
    except:
        urllib.urlretrieve(url, path, reporthook=_dl_progress)

_download(CFG_URL, CFG_NAME)
_download(WEIGHTS_URL, WEIGHTS_NAME)

# Download and Load darknet library
DARKNET_LIB = 'libdarknet.so'
DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
_download(DARKNET_URL, DARKNET_LIB)
DARKNET_LIB = __darknetffi__.dlopen('./' + DARKNET_LIB)
cfg = "./" + str(CFG_NAME)
weights = "./" + str(WEIGHTS_NAME)
net = DARKNET_LIB.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1
print("Converting darknet rnn model to nnvm symbols...")
sym, params = nnvm.frontend.darknet.from_darknet(net, dtype)

# Compile the model on NNVM
data = np.empty([1, net.inputs], dtype)#net.inputs

target = 'llvm'
shape = {'data': data.shape}
print("Compiling the model...")

shape_dict = {'data': data.shape}
dtype_dict = {'data': data.dtype}

with nnvm.compiler.build_config(opt_level=2):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, dtype_dict, params)

# Save the json
def _save_lib():
    '''Save the graph, params and .so to the current directory'''
    print("Saving the compiled output...")
    path_name = 'nnvm_darknet_' + MODEL_NAME
    path_lib = path_name + '_deploy_lib.so'
    lib.export_library(path_lib)
    with open(path_name + "deploy_graph.json", "w") as fo:
        fo.write(graph.json())
    with open(path_name + "deploy_param.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
#_save_lib()

# Execute on TVM
ctx = tvm.cpu(0)

# Create graph runtime
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)

print("RNN generaring text...")

def _proc_rnn_output(out_data):
    '''Generate the characters from the output array'''
    sum_array = 0
    n = out_data.size
    r = random.uniform(0, 1)
    for j in range(n):
        if out_data[j] < .0001:
            out_data[j] = 0
        sum_array += out_data[j]

    for j in range(n):
        out_data[j] *= float(1.0) / sum_array
        r = r - out_data[j]
        if r <= 0:
            return j
    return n-1

def _init_state_memory(rnn_cells_count, dtype):
    '''Initialize memory for states'''
    states = {}
    state_shape = (1024,)
    for i in range(rnn_cells_count):
        k = 'rnn' + str(i) + '_state'
        states[k] = tvm.nd.array(np.zeros(state_shape, dtype).astype(dtype))
    return states

def _set_state_input(runtime, states):
    '''Set the state inputs'''
    for state in states:
        runtime.set_input(state, states[state])

def _get_state_output(runtime, states):
    '''Get the state outputs and save'''
    i = 1
    for state in states:
        data = states[state]
        states[state] = runtime.get_output((i), tvm.nd.empty(data.shape, data.dtype))
        i += 1

out_shape = (net.outputs,)
rnn_cells_count = 3

states = _init_state_memory(rnn_cells_count, dtype)

len_seed = len(seed)
count = len_seed + num
out_txt = ""
random.seed(0)
c = ord(seed[0])
inp_data = np.zeros([net.inputs], dtype)
for i in range(count):
    inp_data[c] = 1
    m.set_input('data', tvm.nd.array(inp_data.astype(dtype)))
    inp_data[c] = 0
    _set_state_input(m, states)
    m.run()
    tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()

    _get_state_output(m, states)
    c = ord(seed[i])  if i < len_seed else _proc_rnn_output(tvm_out)
    out_txt += chr(c)

print("Predicted Text =", out_txt)
