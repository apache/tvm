from collections import namedtuple
from tvm import relay
from tvm.contrib import graph_runtime
import tvm
import numpy as np
import scipy

M, N = 4, 3
feat = np.asmatrix(np.random.rand(M, N))
data    = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
indices = np.array([0,3,1,2,1,2,0,3])
indptr  = np.array([0,2,4,6,8])
csr_mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=(M, M))
print(csr_mat.shape)

#out_np = np.matmul(feat.T, csr_mat.todense())
out_np = np.matmul(csr_mat.T.todense(), feat)
print(out_np)

feat = tvm.nd.array(feat.T.astype('float32'))
data = tvm.nd.array(data.astype('float32'))
indices = tvm.nd.array(indices.astype('int32'))
indptr = tvm.nd.array(indptr.astype('int32'))
adj_dense = tvm.nd.array(csr_mat.todense().astype('float32'))

feat = relay.Constant(feat)
data = relay.Constant(data)
indices = relay.Constant(indices)
indptr = relay.Constant(indptr)
adj_dense = relay.Constant(adj_dense)

Adjacency = namedtuple('Adjacency', ['data', 'indices', 'indptr'])
adj = Adjacency(data, indices, indptr)

output = relay.nn.sparse_dense(feat, adj)
#output = relay.nn.dense(feat, adj_dense)
func = relay.Function(relay.analysis.free_vars(output), output)

# Set the TVM build target
#target = 'cuda -libs=cublas' 
#target = 'cuda -libs=cusparse' 
target = 'cuda'
#target = 'llvm' 

# Build with Relay
with relay.build_config(opt_level=0): 
    graph, lib, params = relay.build_module.build(func, target)

# Generate graph runtime
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)
m.run()
out_tvm = m.get_output(0).asnumpy()
print(np.transpose(out_tvm))
exit()

# Verify the results with the DGL model
tvm.testing.assert_allclose(out_tvm, out_np, atol=1e-3)
