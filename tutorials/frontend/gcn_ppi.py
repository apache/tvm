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

Please refer to DGL doc for DGL installation at
https://docs.dgl.ai/install/index.html

and refer to PyTorch guide for PyTorch installation at
https://pytorch.org/get-started/locally/
"""


######################################################################
# Define GCN in DGL with PyTorch backend
# ------------------
#
# DGL example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
# This part reuses the code from the above example
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 n_infeat,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(n_infeat, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h) 
            # handle api changes for differnt DGL version
            if dgl.__version__ > '0.3':
                h = layer(self.g, h)
            else:
                h = layer(h, self.g)
        return h


######################################################################
# Define the functions to load dataset and evaluate accuracy
# ------------------
# You may substitute this part with your own dataset, here we load data from DGL
import numpy as np
from collections import namedtuple
from sklearn.metrics import f1_score

def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return output, score, loss_data.item()

def evaluate2(logits, labels):
    predict = np.where(logits >= 0.5, 1, 0)
    score = f1_score(labels, predict, average='micro')
    return score


######################################################################
# Load the data and set up model parameters
# ------------------
"""
Parameters
----------
dataset: str
    Name of dataset. You can choose from ['cora', 'citeseer', 'pubmed']. 

num_layer: int
    number of hidden layers

num_hidden: int
    number of the hidden units in the hidden layer

infeat_dim: int
    dimension of the input features

num_classes: int
    dimension of model output (Number of classes)
"""
from dgl.data.ppi import LegacyPPIDataset
from torch.utils.data import DataLoader

batch_size = 2
dropout = 0.5
loss_fcn = torch.nn.BCEWithLogitsLoss()

test_dataset = LegacyPPIDataset(mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

num_layers = 2
num_hidden = 256
num_feats = test_dataset.features.shape[1]
num_classes = test_dataset.labels.shape[1]

######################################################################
# Set up the DGL-PyTorch model and get the golden results
# ------------------
#
# The weights are trained with https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py
#from tvm.contrib.download import download_testdata

g = test_dataset.graph

#target = "cuda: 2"
target = "cpu"
device = torch.device(target)

torch_model = GCN(g,
                  num_feats,
                  num_hidden,
                  num_classes,
                  num_layers,
                  F.relu,
                  dropout)

# Download the pretrained weights
#model_url = "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_%s.torch"%(dataset)
#model_path = download_testdata(model_url, "gcn_%s.pickle"%(dataset), module='gcn_model')

# Load the weights into the model
model_path = "/sampa/home/cyulin/dgl/examples/pytorch/gat/gcn_ppi.torch"
torch_model.load_state_dict(torch.load(model_path))
torch_model = torch_model.to(device)


######################################################################
# Run the DGL model and test for accuracy
# ------------------
def run_torch_model(model, subgraph, feats):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(feats.float())
    return output.numpy()

test_score_list = []
for batch, test_data in enumerate(test_dataloader):
    subgraph, feats, labels = test_data
    feats = feats.to(device)
    torch_logits = run_torch_model(torch_model, subgraph, feats)
    print("Printing first five outputs...\n", torch_logits[0][:5])

    labels = labels.data.cpu().numpy()
    score = evaluate2(torch_logits, labels)
    test_score_list.append(score)
print("Test F1-Score: {:.4f}".format(np.array(test_score_list).mean()))

######################################################################
# Define Graph Convolution Layer in Relay
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
import tvm

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
    Set bias to True to add bias when doing GCN layer

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
# Prepare the parameters needed in the GraphConv layers
# ------------------
# 
import networkx as nx

#def prepare_params(g, data):
def prepare_params(g):
    params = {}
    # Only support float32 as feature for now
    #params['infeats'] = data.features.astype('float32') 

    # Generate adjacency matrix
    adjacency = nx.to_scipy_sparse_matrix(g)
    params['g_data'] = adjacency.data.astype('float32')
    params['indices'] = adjacency.indices.astype('int32')
    params['indptr'] = adjacency.indptr.astype('int32')

    # Normalization w.r.t. node degrees
    degs = [g.in_degree[i] for i in range(g.number_of_nodes())]
    params['norm'] = np.power(degs, -0.5).astype('float32')
    params['norm'] = params['norm'].reshape((params['norm'].shape[0], 1))
    return params

nx_g = test_dataset.nx_graph
params = prepare_params(nx_g)

# Check shape of features and the validity of adjacency matrix
assert params['g_data'] is not None and params['indices'] is not None and params['indptr'] is not None
#assert len(params['infeats'].shape) == 2
#assert params['infeats'].shape[0] == params['indptr'].shape[0] - 1

######################################################################
# Put layers together
# ------------------

# Define input features, norms, adjacency matrix in Relay
infeats = relay.var("infeats", shape=test_dataset.features.shape)
norm = relay.Constant(tvm.nd.array(params['norm']))
g_data = relay.Constant(tvm.nd.array(params['g_data']))
indices = relay.Constant(tvm.nd.array(params['indices']))
indptr = relay.Constant(tvm.nd.array(params['indptr']))

Adjacency = namedtuple('Adjacency', ['data', 'indices', 'indptr'])
adj = Adjacency(g_data, indices, indptr)

# Construct the 2-layer GCN
layers = []
layers.append(GraphConv(
    layer_name="layers.0",
    input_dim=num_feats,
    output_dim=num_hidden,
    adj=adj,
    input=infeats,
    norm=norm,
    activation=relay.nn.relu))
for i in range(1, num_layers):
    layers.append(GraphConv(
        layer_name="layers.{}".format(i),
        input_dim=num_hidden,
        output_dim=num_hidden,
        adj=adj,
        input=layers[-1],
        norm=norm,
        activation=relay.nn.relu))
layers.append(GraphConv(
    layer_name="layers.2",
    input_dim=num_hidden,
    output_dim=num_classes,
    adj=adj,
    input=layers[-1],
    norm=norm,
    activation=None))

# Analyze free variables and generate Relay function
output = layers[-1]
func = relay.Function(relay.analysis.free_vars(output), output)

######################################################################
# Compile and run with TVM
# ------------------
# Export the weigths from PyTorch model to Python Dict
model_params = {}
for param_tensor in torch_model.state_dict():
    model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

for i in range(num_layers+1):
    params["layers.%d.weight"%(i)] = model_params["layers.%d.weight"%(i)]
    params["layers.%d.bias"%(i)] = model_params["layers.%d.bias"%(i)]

# Set the TVM build target
target = 'llvm' # Currently only support `llvm` as target

# Build with Relay
with relay.build_config(opt_level=0): # Currently only support opt_level=0
    graph, lib, params = relay.build(func, target, params=params)

# Generate graph runtime
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)

######################################################################
# Run the TVM model, test for accuracy and verify with DGL
# ------------------
test_score_list = []
for batch, test_data in enumerate(test_dataloader):
    subgraph, feats, labels = test_data
    params['infeats'] = feats.numpy().astype('float32') 
    
    m.set_input(**params)
    m.run()
    tvm_logits = m.get_output(0).asnumpy()
    print("Print the first five outputs...\n", tvm_logits[0][:5])

    labels = labels.data.cpu().numpy()
    score = evaluate2(tvm_logits, labels)
    test_score_list.append(score)
print("Test F1-Score: {:.4f}".format(np.array(test_score_list).mean()))

# Verify the results with the DGL model
tvm.testing.assert_allclose(torch_logits, tvm_logits, atol=1e-3)
