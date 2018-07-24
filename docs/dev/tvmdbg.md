**TVMDBG**

TVM Debugger (TVMDBG) is an interface for debugging TVM's computation graph execution. It helps to provide access to graph structures and tensor values at the TVM runtime.

**Debug Exchange Format**
1. Graph information
     The optimized graph build by nnvm in json format is dumped as it is. This contains the whole information about the graph.
The UX can either use this graph directly or transform this graph to the format UX can understand.

Example of dumped graph:
```
{
  "nodes": [                                    # List of nodes
    {
      "op": "null",                             # operation type = null, this is a placeholder/variable/input node
      "name": "x",                              # Name of the argument node
      "inputs": []                              # inputs for this node, its none since this is an argument node
    },
    {
      "op": "tvm_op",                           # operation type = tvm_op, this node can be executed
      "name": "relu0",                          # Name of the node
      "attrs": {                                # Attributes of the node
        "flatten_data": "0",                    # Whether this data need to be flattened
        "func_name": "fuse_l2_normalize_relu",  # Fused function name, this function is there in the lib
        "num_inputs": "1",                      # Number of inputs for this node
        "num_outputs": "1"                      # Number of outputs this node produces
      },
      "inputs": [[0, 0, 0]]                     # Position of the inputs for this operation
    }
  ],
  "arg_nodes": [0],                             # Which all nodes in this are argument nodes
  "node_row_ptr": [0, 1, 2],                    # Row indices for faster depth first search
  "heads": [[1, 0, 0]],                         # Position of the output nodes for this operation
  "attrs": {                                    # Attributes for the graph
    "storage_id": ["list_int", [1, 0]],         # memory slot id for each node in the storage layout
    "dtype": ["list_int", [0, 0]],              # Datatype of each node (enum value)
    "dltype": ["list_str", [                    # Datatype of each node in order
        "float32",
        "float32"]],
    "shape": ["list_shape", [                   # Shape of each node k order
        [1, 3, 20, 20],
        [1, 3, 20, 20]]]
  }
}

```

2. Tensor dumping
     The tensor received after execution is in `tvm.ndarray` type. But the UX cannot read this format. So it can be transformed to numpy format using `asnumpy()`.
More about numpy format can be read in the below link.
[numpy-format](https://docs.scipy.org/doc/numpy/neps/npy-format.html)
Each node in the graph will be dumped to individual files, in the dump folder. These files will be created after execution of each node.


**How to use TVMDBG?**
1. In `config.cmake` set the `USE_GRAPH_RUNTIME_DEBUG` flag to `ON`
```
# Whether enable additional graph debug functions
set(USE_GRAPH_RUNTIME_DEBUG ON)
```
2. Do 'make' tvm, so that it will make the `libtvm_runtime.so`

3. In frontend script file instead of `from tvm.contrib import graph_runtime` import the `debug_runtime` `from tvm.contrib.debugging import debug_runtime as graph_runtime`

```
#from tvm.contrib import graph_runtime
from tvm.contrib.debugging import debug_runtime as graph_runtime
m = graph_runtime.create(graph, lib, ctx, dump_root="/tmp/tvmdbg")
# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
m.run()
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
```

The outputs are dumped to a temporary folder in `/tmp` folder or the folder specified while creating the runtime.
