import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h


from dgl.data import load_data
from collections import namedtuple
import numpy as np

def load_dataset(dataset="cora"):
    args = namedtuple("args", ["dataset"])
    data = load_data(args(dataset))

    # Remove self-loops to avoid duplicate passing of a node's feature to itself
    g = data.graph
    g.add_edges_from(zip(g.nodes, g.nodes))

    mean_g = g
    mean_g.remove_edges_from(g.selfloop_edges())
    mean_adj = nx.to_numpy_array(mean_g)
    #mean_adj = nx.to_numpy_matrix(mean_g)
    #print(mean_adj.shape)
    #print(mean_adj[0, [8,14,258,435,544]])
    #print(mean_g.edges(0))
    #print(np.nonzero(mean_adj[0]))
    adj_cnt = np.sum(mean_adj, axis=1)
    avg = np.reciprocal(adj_cnt)
    mean_adj = (mean_adj.T * avg).T

    return g, data, mean_adj


def evaluate(data, logits):
    test_mask = data.test_mask # the test set which isn't included in the training phase

    pred = logits.argmax(axis=1)
    acc = ((pred == data.labels) * test_mask).sum() / test_mask.sum()

    return acc


dataset = "cora"

g, data, mean_adj = load_dataset(dataset)

num_layers = 1
num_hidden = 16
infeat_dim = data.features.shape[1]
num_classes = data.num_labels

#from tvm.contrib.download import download_testdata
from dgl import DGLGraph
#
features = torch.FloatTensor(data.features)
dgl_g = DGLGraph(g)
#
torch_model = GraphSAGE(dgl_g,
                  infeat_dim,
                  num_hidden,
                  num_classes,
                  num_layers,
                  F.relu,
                  0.5,
                  "mean")
#
## Download the pretrained weights
#model_url = "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_%s.torch"%(dataset)
#model_path = download_testdata(model_url, "gcn_%s.pickle"%(dataset), module='gcn_model')
model_path = "/sampa/home/cyulin/dgl/examples/pytorch/graphsage/graphsage_cora.torch"
#
## Load the weights into the model
torch_model.load_state_dict(torch.load(model_path))
#
#
torch_model.eval()
with torch.no_grad():
    logits_torch = torch_model(features)
print("Print the first five outputs from DGL-PyTorch execution\n", logits_torch[:5])

acc = evaluate(data, logits_torch.numpy())
print("Test accuracy of DGL results: {:.2%}".format(acc))

from tvm import relay
from tvm.contrib import graph_runtime
import tvm

def SageConv(layer_name,
              input_dim,
              output_dim,
              adj,
              mean_adj,
              input,
              norm=None,
              bias=True,
              activation=None):
    #if norm is not None:
    #    input = relay.multiply(input, norm)

    self_w = relay.var(layer_name + ".fc_self.weight", shape=(input_dim, output_dim))
    self_o = relay.nn.dense(input, self_w)
    if bias is True:
        self_bias = relay.var(layer_name + ".fc_self.bias", shape=(output_dim, 1))
        self_o = relay.nn.bias_add(self_o, self_bias, axis=-1)
    
    input_t = relay.transpose(input)
    neigh_feat = relay.nn.dense(input_t, mean_adj)
    neigh_feat = relay.transpose(neigh_feat)

    neigh_w = relay.var(layer_name + ".fc_neigh.weight", shape=(input_dim, output_dim))
    neigh_o = relay.nn.dense(neigh_feat, neigh_w)
    if bias is True:
        neigh_bias = relay.var(layer_name + ".fc_neigh.bias", shape=(output_dim, 1))
        neigh_o = relay.nn.bias_add(neigh_o, neigh_bias, axis=-1)

    output = relay.add(self_o, neigh_o)

    if activation is not None:
        output = activation(output)
    #if norm is not None:
    #    output = relay.multiply(output, norm)
    return output

def prepare_params(g, data):
    params = {}
    params['infeats'] = data.features.astype('float32') # Only support float32 as feature for now

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

params = prepare_params(g, data)

# Check shape of features and the validity of adjacency matrix
assert len(params['infeats'].shape) == 2
assert params['g_data'] is not None and params['indices'] is not None and params['indptr'] is not None
assert params['infeats'].shape[0] == params['indptr'].shape[0] - 1

infeats = relay.var("infeats", shape=data.features.shape)
norm = relay.Constant(tvm.nd.array(params['norm']))
g_data = relay.Constant(tvm.nd.array(params['g_data']))
indices = relay.Constant(tvm.nd.array(params['indices']))
indptr = relay.Constant(tvm.nd.array(params['indptr']))

Adjacency = namedtuple('Adjacency', ['data', 'indices', 'indptr'])
adj = Adjacency(g_data, indices, indptr)

mean_adj = relay.Constant(tvm.nd.array(mean_adj.astype('float32'))) 

# Construct the 2-layer GCN
layers = []
layers.append(SageConv(
    layer_name="layers.0",
    input_dim=infeat_dim,
    output_dim=num_hidden,
    adj=adj,
    mean_adj=mean_adj,
    input=infeats,
    norm=norm,
    activation=relay.nn.relu
))
layers.append(SageConv(
    layer_name="layers.1",
    input_dim=num_hidden,
    output_dim=num_classes,
    adj=adj,
    mean_adj=mean_adj,
    input=layers[-1],
    norm=norm,
    activation=None
))

# Analyze free variables and generate Relay function
output = layers[-1]
func = relay.Function(relay.analysis.free_vars(output), output)
#mod = relay.Module.from_expr(func)
#mod = relay.transform.InferType()(mod)
#print(mod.astext(show_meta_data=False))

model_params = {}
for param_tensor in torch_model.state_dict():
    model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

for i in range(num_layers+1):
    params["layers.%d.fc_self.weight"%(i)] = model_params["layers.%d.fc_self.weight"%(i)]
    params["layers.%d.fc_self.bias"%(i)] = model_params["layers.%d.fc_self.bias"%(i)]
    params["layers.%d.fc_neigh.weight"%(i)] = model_params["layers.%d.fc_neigh.weight"%(i)]
    params["layers.%d.fc_neigh.bias"%(i)] = model_params["layers.%d.fc_neigh.bias"%(i)]

# Set the TVM build target
target = 'llvm' # Currently only support `llvm` as target

# Build with Relay
with relay.build_config(opt_level=0): # Currently only support opt_level=0
    graph, lib, params = relay.build(func, target, params=params)

# Generate graph runtime
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)

m.run()
logits_tvm = m.get_output(0).asnumpy()
print("Print the first five outputs from TVM execution\n", logits_tvm[:5])

labels = data.labels
test_mask = data.test_mask

acc = evaluate(data, logits_tvm)
print("Test accuracy of TVM results: {:.2%}".format(acc))

# Verify the results with the DGL model
tvm.testing.assert_allclose(logits_torch, logits_tvm, atol=1e-3)
