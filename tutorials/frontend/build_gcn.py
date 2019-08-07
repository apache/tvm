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
**Author**: `Yulun Yao <https://yulunyao.io/>`_

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

def GraphConv(
            layer_name,
            input_dim,
            output_dim,
            adj,
            input,
            activation=None,
            norm=None,
            ):
    r"""
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

    activation: <function relay.op.nn>,
    Activation function applies to the output. e.g. relay.nn.{relu, sigmoid, log_softmax, softmax, leaky_relu}


    Returns
    ----------
    output: tvm.relay.Expr
    The Output Tensor for this layer [num_nodes, output_dim]
    """
    if norm is not None:
        input = relay.multiply(input, norm)
    weight = relay.var(layer_name + "_weight", shape=(input_dim, output_dim))
    weight_transposed = relay.transpose(weight)
    dense = relay.nn.dense(weight_transposed, input)
    output = relay.nn.sparse_dense(dense, adj)
    output_transposed = relay.transpose(output)
    if norm is not None:
        output_transposed = relay.multiply(output_transposed, norm)
    if activation is not None:
        output_transposed = activation(output_transposed)
    return output_transposed

######################################################################
# Load the dataset
# ------------------
# You may substitute this part with your own dataset, here we load data from DGL to benchmark
import tvm, dgl, scipy
import numpy as np
import networkx as nx
from collections import namedtuple
from dgl.data import load_data

def load_dataset(dataset="cora"):
    args = namedtuple("args", ["dataset"])
    dataset = load_data(args(dataset))

    params = {}
    params['infeats'] = dataset.features.astype('float32') # Only support float32 as feature for now

    # Remove self-loops to avoid duplicate passing of a node's feature to itself
    g = dataset.graph
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

    return params

######################################################################
# Set up model Parameters
# ------------------

r"""
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

num_hidden = 1
hidden_dim = 16
num_classes = 7
target = 'llvm'
activation = relay.nn.relu

dataset = "cora"
params = load_dataset(dataset)

# Check shape of features
assert len(params['infeats'].shape) == 2
nnodes, input_dim = params['infeats'].shape

# Check validity of adjacency matrix
assert params['data'] is not None and params['indices'] is not None and params['indptr'] is not None
assert nnodes == params['indptr'].shape[0] - 1

######################################################################
# Put layers together
# ------------------

layers = []

# Define input features, norms, adjacency matrix
infeats = relay.var("infeats", shape=(nnodes, input_dim))

norm = relay.Constant(tvm.nd.array(params['norm']))

data = relay.Constant(tvm.nd.array(params['data']))
indices = relay.Constant(tvm.nd.array(params['indices']))
indptr = relay.Constant(tvm.nd.array(params['indptr']))

Adjacency = namedtuple('Adjacency', ['data', 'indices', 'indptr'])
adj = Adjacency(data, indices, indptr)

# Generate Input Layer
layers.append(GraphConv(
    layer_name= 'in',
    input_dim= input_dim,
    output_dim= hidden_dim,
    adj = adj,
    input= infeats,
    activation= activation,
    norm= norm,
))

# Generate Hidden Layers
for i in range(num_hidden):
    layers.append(GraphConv(
        layer_name= str(i),
        input_dim= hidden_dim,
        output_dim= hidden_dim,
        adj = adj,
        input= layers[-1],
        activation= activation,
        norm= norm,
    ))

# Generate Output Layer
layers.append(GraphConv(
    layer_name= 'out',
    input_dim= hidden_dim,
    output_dim= num_classes,
    adj = adj,
    input= layers[-1],
    activation= activation,
    norm= norm,
))
output = layers[-1]

# Analyze free variables and generate function
func = relay.Function(relay.analysis.free_vars(output), output)

######################################################################
# Compile and run
# ------------------
# We achieved 6.5x speedup for this dataset against dgl given the same model parameters.
# Output numerical difference < 10e-4 %.
#
# DGL version: https://github.com/dmlc/dgl/blob/master/examples/mxnet/gcn/gcn.py
from tvm.contrib import graph_runtime
import time

# Set up weights. You can modify this part and use your own trained weights.
params['in_weight'] = np.ones((input_dim, hidden_dim), dtype='float32')
params['out_weight'] = np.ones((hidden_dim, num_classes), dtype='float32')
for i in range(num_hidden):
    params["%s_weight"%(str(i))] = np.ones((hidden_dim, hidden_dim), dtype='float32')

# Generate graph and library
with relay.build_config(opt_level=0): # Currently only support opt_level=0
    graph, lib, params = relay.build(func, target, params=params)
    lib.save("lib.o")

# Generate module for llvm
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)

print("finished compiling, testing inference time cost")
totaltime = 0
for i in range(30):
    st = time.time()
    # One forward pass on the entire network
    m.run()
    end = time.time()
    # Retrieve output Tensor as numpy array
    outval = m.get_output(0).asnumpy()

    totaltime += (end-st)

    if i == 0:
        print("features of first five nodes \n %s" % outval[:5])
    if i == 4:
        print("5 Cycle Average Forward Pass Time ", totaltime/5)
print("30 Cycle Average Forward Pass Time ", totaltime/30)
