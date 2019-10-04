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
Building a Graph Convolutional Network
=====================
**Author**: `Yulun Yao <https://yulunyao.io/>`_, \
            `Chien-Yu Lin <https://homes.cs.washington.edu/~cyulin/>`_

This article is an introductory tutorial to build a Graph Convolutional Network (GCN) with Relay.

In this tutorial, we will run our GCN on Cora dataset to demonstrate.

Cora dataset is a common benchmark for Graph Neural Networks (GNN) and frameworks that support GNN training and inference.

We directly load the dataset from DGL library to do the apples to apples comparison against DGL.

Please refer to DGL tutorial on installation at
https://docs.dgl.ai/install/index.html

GPU support and more sparse operators will soon follow.
"""

######################################################################
# Define Graph Convolution Layer
# ----------------------------
# To run GCN on TVM, we first need to implement Graph Convolution Layer.
#
# You may refer to https://github.com/dmlc/dgl/blob/master/python/dgl/nn/mxnet/conv.py for a GraphConv Layer implemented in DGL with MXNet Backend
#
# The layer is defined with below operations, note that we apply two transposes to keep adjacency matrix on right hand side of sparse_dense operator,
# this method is temporary and will be updated in next few weeks when we have sparse matrix transpose and support for left sparse operator.
#
#  .. math::
#
#            \mbox{GraphConv}(A, H, W)   = A * H * W
#                                        = ((H * W)^t * A^t)^t
#                                        = ((W^t * H^t) * A^t)^t
from tvm import relay
from tvm.contrib import graph_runtime
import tvm, dgl, scipy
import numpy as np
import networkx as nx
from collections import namedtuple
from dgl.data import load_data

from tvm.contrib.download import download_testdata
import pickle

def GraphConv(layer_name,
              input_dim,
              output_dim,
              adj,
              input,
              norm=None,
              bias=True,
              activation=None):
    """
    Parameters
    ----------
    layer_name: str
    Name of layer

    input_dim: int
    Input dimension per node feature

    output_dim: int,
    Output dimension per node feature

    adj: namedtuple,
    Graph representation (Adjacency Matrix) in Sparse Format (`data`, `indices`, `indptr`),
    where `data` has shape [num_nonzeros], indices` has shape [num_nonzeros], `indptr` has shape [num_nodes + 1]

    input: relay.Expr,
    Input feature to current layer with shape [num_nodes, input_dim]

    norm: relay.Expr,
    Norm passed to this layer to normalize features before and after Convolution.

    bias: bool
    Set bias to True to add bias when doing gcn layer

    activation: <function relay.op.nn>,
    Activation function applies to the output. e.g. relay.nn.{relu, sigmoid, log_softmax, softmax, leaky_relu}

    Returns
    ----------
    output: tvm.relay.Expr
    The Output Tensor for this layer [num_nodes, output_dim]
    """
    if norm is not None:
        input = relay.multiply(input, norm)

    weight = relay.var(layer_name + ".weight", shape=(input_dim, output_dim))
    weight_t = relay.transpose(weight)
    dense = relay.nn.dense(weight_t, input)
    output = relay.nn.sparse_dense(dense, adj)
    output_t = relay.transpose(output)
    if norm is not None:
        output_t = relay.multiply(output_t, norm)
    if bias is True:
        _bias = relay.var(layer_name + ".bias", shape=(output_dim, 1))
        output_t = relay.nn.bias_add(output_t, _bias, axis=-1)
    if activation is not None:
        output_t = activation(output_t)
    return output_t

######################################################################
# Load the dataset
# ------------------
# You may substitute this part with your own dataset, here we load data from DGL to benchmark

def load_dataset(dataset="cora"):
    args = namedtuple("args", ["dataset"])
    data = load_data(args(dataset))

    params = {}
    params['infeats'] = data.features.astype('float32') # Only support float32 as feature for now

    # Remove self-loops to avoid duplicate passing of a node's feature to itself
    g = data.graph
    g.remove_edges_from(g.selfloop_edges())
    g.add_edges_from(zip(g.nodes, g.nodes))

    # Generate adjacency matrix
    adjacency = nx.to_scipy_sparse_matrix(g)
    params['data'] = adjacency.data.astype('float32')
    params['indices'] = adjacency.indices.astype('int32')
    params['indptr'] = adjacency.indptr.astype('int32')

    # Normalization w.r.t. node degrees
    degs = [g.in_degree[i] for i in range(g.number_of_nodes())]
    params['norm'] = np.power(degs, -0.5).astype('float32')
    params['norm'] = params['norm'].reshape((params['norm'].shape[0], 1))

    return data, params

######################################################################
# Set up model Parameters
# ------------------

"""
Parameters
----------
num_hidden: int
    number of hidden layers

hidden_dim: int
    input dimension of hidden layers

num_classes: int
    dimension of model output (Number of classes)

target: str
    currently only support llvm, GPU support will be added in next few weeks

activation: <function relay.op.nn>,
    Activation function applied to the output. e.g. relay.nn.{relu, sigmoid, log_softmax, softmax, leaky_relu}

dataset: str
    Name of dataset. You can pick from ['cora', 'citeseer', 'pubmed'] or you can use your own.
"""

dataset = "cora"
data, params = load_dataset(dataset)

num_hidden = 1
hidden_dim = [16]
num_classes = data.num_labels
bias = True
test_mask = data.test_mask
labels = data.labels
target = 'llvm'
activation = relay.nn.relu

# Check shape of features
assert len(params['infeats'].shape) == 2
nnodes, input_dim = params['infeats'].shape

# Check validity of adjacency matrix
assert params['data'] is not None and params['indices'] is not None and params['indptr'] is not None
assert nnodes == params['indptr'].shape[0] - 1

######################################################################
# Put layers together
# ------------------

# Define input features, norms, adjacency matrix
infeats = relay.var("infeats", shape=(nnodes, input_dim))

norm = relay.Constant(tvm.nd.array(params['norm']))

data = relay.Constant(tvm.nd.array(params['data']))
indices = relay.Constant(tvm.nd.array(params['indices']))
indptr = relay.Constant(tvm.nd.array(params['indptr']))

Adjacency = namedtuple('Adjacency', ['data', 'indices', 'indptr'])
adj = Adjacency(data, indices, indptr)

# Construct a 2-layer GCN
layers = []

layers.append(GraphConv(
    layer_name="layers.0",
    input_dim=input_dim,
    output_dim=hidden_dim[0],
    adj=adj,
    input=infeats,
    norm=norm,
    bias=bias,
    activation=activation
))

layers.append(GraphConv(
    layer_name="layers.1",
    input_dim=hidden_dim[0],
    output_dim=num_classes,
    adj=adj,
    input=layers[-1],
    norm=norm,
    bias=bias,
    activation=activation
))

output = layers[-1]

# Analyze free variables and generate function
func = relay.Function(relay.analysis.free_vars(output), output)

######################################################################
# Compile and run
# ------------------
#
# DGL version: https://github.com/dmlc/dgl/blob/master/examples/mxnet/gcn/gcn.py

# Download pretrained GCN model
model_url = "https://homes.cs.washington.edu/~cyulin/media/gcn_%s.pickle"%(dataset)
model_path = download_testdata(model_url, 'gcn.pickle', module='gcn_model')

with open(model_path, 'rb') as fp:
    model_params = pickle.load(fp)

for i in range(num_hidden+1):
    params["layers.%d.weight"%(i)] = model_params["layers.%d.weight"%(i)]
    params["layers.%d.bias"%(i)] = model_params["layers.%d.bias"%(i)]

# Build with relay
with relay.build_config(opt_level=0): # Currently only support opt_level=0
    graph, lib, params = relay.build(func, target, params=params)
    lib.save("lib.o")

# Generate graph runtime
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)

# Run the model for one time and test for accuracy
m.run()
outval = m.get_output(0).asnumpy()
pred = outval.argmax(axis=1)
accuracy = ((pred == labels) * test_mask).sum() / test_mask.sum()
print("Test accuracy {:.2%}".format(accuracy))
print(outval[:5])

# Evaluate the runtime
print("Evaluate inference time cost...")
timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
tcost = timer()
prof_res = tcost.results
prof_res = np.array(tcost.results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.6f ms (%.6f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
