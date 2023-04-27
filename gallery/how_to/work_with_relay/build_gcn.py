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
======================================
**Author**: `Yulun Yao <https://yulunyao.io/>`_, \
            `Chien-Yu Lin <https://homes.cs.washington.edu/~cyulin/>`_

This article is an introductory tutorial to build a Graph Convolutional Network (GCN) with Relay.
In this tutorial, we will run our GCN on Cora dataset to demonstrate.
Cora dataset is a common benchmark for Graph Neural Networks (GNN) and frameworks that support GNN training and inference.
We directly load the dataset from DGL library to do the apples to apples comparison against DGL.

.. code-block:: bash

    %%shell
    pip install torch==2.0.0
    pip install dgl==v1.0.0

Please refer to DGL doc for installation at
https://docs.dgl.ai/install/index.html.

Please refer to PyTorch guide for PyTorch installation at
https://pytorch.org/get-started/locally/.
"""


######################################################################
# Define GCN in DGL with PyTorch backend
# --------------------------------------
#
# DGL example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
# This part reuses the code from the above example.
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, g, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(n_infeat, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            if dgl.__version__ > "0.3":
                h = layer(self.g, h)
            else:
                h = layer(h, self.g)
        return h


######################################################################
# Define the functions to load dataset and evaluate accuracy
# ----------------------------------------------------------
# You may substitute this part with your own dataset, here we load data from DGL
from dgl.data import load_data
from collections import namedtuple


def evaluate(g, logits):
    label = g.ndata["label"]
    test_mask = g.ndata["test_mask"]

    pred = logits.argmax(axis=1)
    acc = (torch.Tensor(pred[test_mask]) == label[test_mask]).float().mean()

    return acc


######################################################################
# Load the data and set up model parameters
# -----------------------------------------
"""
Parameters
----------
num_layer: int
    number of hidden layers

num_hidden: int
    number of the hidden units in the hidden layer

infeat_dim: int
    dimension of the input features

num_classes: int
    dimension of model output (Number of classes)
"""

dataset = dgl.data.CoraGraphDataset()
dgl_g = dataset[0]
num_layers = 1
num_hidden = 16
features = dgl_g.ndata["feat"]
infeat_dim = features.shape[1]
num_classes = dataset.num_classes

######################################################################
# Set up the DGL-PyTorch model and get the golden results
# -------------------------------------------------------
#
# The weights are trained with https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py
from tvm.contrib.download import download_testdata

features = torch.FloatTensor(features)

torch_model = GCN(dgl_g, infeat_dim, num_hidden, num_classes, num_layers, F.relu)

# Download the pretrained weights
model_url = "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_cora.torch"
model_path = download_testdata(model_url, "gcn_cora.pickle", module="gcn_model")

# Load the weights into the model
torch_model.load_state_dict(torch.load(model_path))


######################################################################
# Run the DGL model and test for accuracy
# ---------------------------------------
torch_model.eval()
with torch.no_grad():
    logits_torch = torch_model(features)
print("Print the first five outputs from DGL-PyTorch execution\n", logits_torch[:5])

acc = evaluate(dgl_g, logits_torch.numpy())
print("Test accuracy of DGL results: {:.2%}".format(acc))


######################################################################
# Define Graph Convolution Layer in Relay
# ---------------------------------------
# To run GCN on TVM, we first need to implement Graph Convolution Layer.
# You may refer to https://github.com/dmlc/dgl/blob/master/python/dgl/nn/mxnet/conv/graphconv.py for a GraphConv Layer implemented in DGL with MXNet Backend
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
from tvm.contrib import graph_executor
import tvm
from tvm import te


def GraphConv(layer_name, input_dim, output_dim, adj, input, norm=None, bias=True, activation=None):
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
# -----------------------------------------------------
#
import numpy as np
import networkx as nx


def prepare_params(g):
    params = {}
    params["infeats"] = g.ndata["feat"].numpy().astype("float32")

    # Generate adjacency matrix
    nx_graph = dgl.to_networkx(g)
    adjacency = nx.to_scipy_sparse_array(nx_graph)
    params["g_data"] = adjacency.data.astype("float32")
    params["indices"] = adjacency.indices.astype("int32")
    params["indptr"] = adjacency.indptr.astype("int32")

    # Normalization w.r.t. node degrees
    degs = [g.in_degrees(i) for i in range(g.number_of_nodes())]
    params["norm"] = np.power(degs, -0.5).astype("float32")
    params["norm"] = params["norm"].reshape((params["norm"].shape[0], 1))

    return params


params = prepare_params(dgl_g)

# Check shape of features and the validity of adjacency matrix
assert len(params["infeats"].shape) == 2
assert (
    params["g_data"] is not None and params["indices"] is not None and params["indptr"] is not None
)
assert params["infeats"].shape[0] == params["indptr"].shape[0] - 1

######################################################################
# Put layers together
# -------------------

# Define input features, norms, adjacency matrix in Relay
infeats = relay.var("infeats", shape=features.shape)
norm = relay.Constant(tvm.nd.array(params["norm"]))
g_data = relay.Constant(tvm.nd.array(params["g_data"]))
indices = relay.Constant(tvm.nd.array(params["indices"]))
indptr = relay.Constant(tvm.nd.array(params["indptr"]))

Adjacency = namedtuple("Adjacency", ["data", "indices", "indptr"])
adj = Adjacency(g_data, indices, indptr)

# Construct the 2-layer GCN
layers = []
layers.append(
    GraphConv(
        layer_name="layers.0",
        input_dim=infeat_dim,
        output_dim=num_hidden,
        adj=adj,
        input=infeats,
        norm=norm,
        activation=relay.nn.relu,
    )
)
layers.append(
    GraphConv(
        layer_name="layers.1",
        input_dim=num_hidden,
        output_dim=num_classes,
        adj=adj,
        input=layers[-1],
        norm=norm,
        activation=None,
    )
)

# Analyze free variables and generate Relay function
output = layers[-1]

######################################################################
# Compile and run with TVM
# ------------------------
#
# Export the weights from PyTorch model to Python Dict
model_params = {}
for param_tensor in torch_model.state_dict():
    model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

for i in range(num_layers + 1):
    params["layers.%d.weight" % (i)] = model_params["layers.%d.weight" % (i)]
    params["layers.%d.bias" % (i)] = model_params["layers.%d.bias" % (i)]

# Set the TVM build target
target = "llvm"  # Currently only support `llvm` as target

func = relay.Function(relay.analysis.free_vars(output), output)
func = relay.build_module.bind_params_by_name(func, params)
mod = tvm.IRModule()
mod["main"] = func
# Build with Relay
with tvm.transform.PassContext(opt_level=0):  # Currently only support opt_level=0
    lib = relay.build(mod, target, params=params)

# Generate graph executor
dev = tvm.device(target, 0)
m = graph_executor.GraphModule(lib["default"](dev))

######################################################################
# Run the TVM model, test for accuracy and verify with DGL
# --------------------------------------------------------
m.run()
logits_tvm = m.get_output(0).numpy()
print("Print the first five outputs from TVM execution\n", logits_tvm[:5])

acc = evaluate(dgl_g, logits_tvm)
print("Test accuracy of TVM results: {:.2%}".format(acc))

import tvm.testing

# Verify the results with the DGL model
tvm.testing.assert_allclose(logits_torch, logits_tvm, atol=1e-3)
