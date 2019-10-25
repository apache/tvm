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
import time

def load_dataset(dataset="cora"):
    args = namedtuple("args", ["dataset"])
    data = load_data(args(dataset))

    # Remove self-loops to avoid duplicate passing of a node's feature to itself
    g = data.graph
    g.remove_edges_from(g.selfloop_edges())

    adj = nx.to_numpy_array(g)

    return g, adj, data


def evaluate(data, logits):
    test_mask = data.test_mask # the test set which isn't included in the training phase

    pred = logits.argmax(axis=1)
    acc = ((pred == data.labels) * test_mask).sum() / test_mask.sum()
    return acc

def time_evaluator(model, input, times):
    model.eval()
    time_costs = []
    with torch.no_grad():
        for i in range(times):
            start = time.time()
            logits= model(input)
            if i > 0:
                time_costs.append(time.time()-start)
    return time_costs


dataset = "cora"

g, adj, data = load_dataset(dataset)

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

torch_model.eval()
with torch.no_grad():
    logits_torch = torch_model(features)
print("Print the first five outputs from DGL-PyTorch execution\n", logits_torch[:5])

acc = evaluate(data, logits_torch.numpy())
print("Test accuracy of DGL results: {:.2%}".format(acc))

time_costs = time_evaluator(torch_model, features, 10)
prof_res = np.array(time_costs) * 1000  # convert to millisecond
print("DGL mean inference time (std dev): %.6f ms (%.6f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


from tvm import relay
from tvm.contrib import graph_runtime
import tvm

def mean_aggregator(feats, adj):
    adj_sum = relay.sum(adj, axis=1)
    adj_sum = relay.transpose(adj_sum)
    feats = relay.transpose(feats)
    agg_feats = relay.nn.dense(feats, adj)
    agg_feats = relay.divide(agg_feats, adj_sum)
    agg_feats = relay.transpose(agg_feats)
    return agg_feats

def SageConv(layer_name,
              input_dim,
              output_dim,
              adj,
              input,
              norm=None,
              bias=True,
              activation=None):
    neigh_feat = mean_aggregator(input, adj)

    neigh_w = relay.var(layer_name + ".fc_neigh.weight", shape=(input_dim, output_dim))
    neigh_o = relay.nn.dense(neigh_feat, neigh_w)

    self_w = relay.var(layer_name + ".fc_self.weight", shape=(input_dim, output_dim))
    self_o = relay.nn.dense(input, self_w)
    if bias is True:
        self_bias = relay.var(layer_name + ".fc_self.bias", shape=(output_dim, 1))
        self_o = relay.nn.bias_add(self_o, self_bias, axis=-1)
        neigh_bias = relay.var(layer_name + ".fc_neigh.bias", shape=(output_dim, 1))
        neigh_o = relay.nn.bias_add(neigh_o, neigh_bias, axis=-1)

    output = relay.add(self_o, neigh_o)

    if activation is not None:
        output = activation(output)
    return output


infeats = relay.var("infeats", shape=data.features.shape)

adj = relay.Constant(tvm.nd.array(adj.astype('float32'))) 

# Construct the 2-layer GCN
layers = []
layers.append(SageConv(
    layer_name="layers.0",
    input_dim=infeat_dim,
    output_dim=num_hidden,
    adj=adj,
    input=infeats,
    activation=relay.nn.relu
))
layers.append(SageConv(
    layer_name="layers.1",
    input_dim=num_hidden,
    output_dim=num_classes,
    adj=adj,
    input=layers[-1],
    activation=None
))

# Analyze free variables and generate Relay function
output = layers[-1]
func = relay.Function(relay.analysis.free_vars(output), output)

params = {}
params['infeats'] = data.features.astype('float32') # Only support float32 as feature for now

for param_tensor in torch_model.state_dict():
    params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

# Check shape of features and the validity of adjacency matrix
assert len(params['infeats'].shape) == 2

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

timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
tcost = timer()
prof_res = tcost.results
prof_res = np.array(tcost.results) * 1000  # convert to millisecond
print("TVM mean inference time (std dev): %.6f ms (%.6f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
