# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Darknet Models for RNN
==============================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

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
import numpy as np
import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from nnvm.testing.darknet import __darknetffi__
import nnvm
import nnvm.frontend.darknet

# Set the parameters
# -----------------------
# Set the seed value and the number of characters to predict

#Model name
MODEL_NAME = 'rnn'
#Seed value
seed = 'Thus'
#Number of characters to predict
num = 1000

# Download required files
# -----------------------
# Download cfg and weights file if first time.
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
WEIGHTS_URL = REPO_URL + 'weights/' + WEIGHTS_NAME + '?raw=true'

cfg_path = download_testdata(CFG_URL, CFG_NAME, module='darknet')
weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module='darknet')

# Download and Load darknet library
DARKNET_LIB = 'libdarknet.so'
DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module='darknet')
DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1

# Import the graph to NNVM
# ------------------------
# Import darknet graph definition to nnvm.
#
# Results:
#   sym: nnvm graph for rnn model
#   params: params converted from darknet weights
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

# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the NNVM compiled model on cpu target.

# Set the cpu context
ctx = tvm.cpu(0)
# Create graph runtime
m = graph_runtime.create(graph, lib, ctx)
# Set the params to runtime
m.set_input(**params)

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

def _proc_rnn_output(out_data):
    '''Generate the characters from the output array'''
    sum_array = 0
    n = out_data.size
    r = random.uniform(0, 1)
    for j in range(n):
        if out_data[j] < 0.0001:
            out_data[j] = 0
        sum_array += out_data[j]

    for j in range(n):
        out_data[j] *= float(1.0) / sum_array
        r = r - out_data[j]
        if r <= 0:
            return j
    return n-1

print("RNN generaring text...")

out_shape = (net.outputs,)
rnn_cells_count = 3

# Initialize state memory
# -----------------------
states = _init_state_memory(rnn_cells_count, dtype)

len_seed = len(seed)
count = len_seed + num
out_txt = ""

#Initialize random seed
random.seed(0)
c = ord(seed[0])
inp_data = np.zeros([net.inputs], dtype)

# Run the model
# -------------

# Predict character by character till `num`
for i in range(count):
    inp_data[c] = 1

    # Set the input data
    m.set_input('data', tvm.nd.array(inp_data.astype(dtype)))
    inp_data[c] = 0

    # Set the state inputs
    _set_state_input(m, states)

    # Run the model
    m.run()

    # Get the output
    tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()

    # Get the state outputs
    _get_state_output(m, states)

    # Get the predicted character and keep buffering it
    c = ord(seed[i])  if i < len_seed else _proc_rnn_output(tvm_out)
    out_txt += chr(c)

print("Predicted Text =", out_txt)
