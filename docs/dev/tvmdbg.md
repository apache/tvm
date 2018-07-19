**TVMDBG**

TVM Debugger (TVMDBG) is a UI based specialized debugger for TVM's computation graphs. It provides access to internal graph structures and tensor values at TVM runtime.

**Why**  **TVMDBG**
In TVM's current computation-graph framework, almost all actual computation after graph construction happens in a single Python function. Basic Python debugging tools such as pdb cannot be used to debug tvm.run, due to the fact that TVM's graph execution happens in the underlying C++ layer. C++ debugging tools such as gdb are not ideal either, because of their inability to recognize and organize the stack frames and variables in a way relevant to TVM's operations, tensors and other graph constructs.

TVMDBG addresses these limitations. Among the features provided by TVMDBG, the following ones are designed to facilitate runtime debugging of TVM:
- Inspection of runtime ops output values and node connections
- Time consumed by each fused operation

**Note:** The TVM debugger uses a curses-based text user interface.

**How to use TVMDBG?**
1. In `config.cmake` set the `USE_GRAPH_RUNTIME_DEBUG` flag to `ON`
```
# Whether enable additional graph debug functions
set(USE_GRAPH_RUNTIME_DEBUG ON)
```
2. Make tvm so that it will make the `libtvm_runtime.so`

3. In the graph build file instead of `from tvm.contrib import graph_runtime` import the `debug_runtime` `from tvm.contrib.debugging import debug_runtime as graph_runtime`
```
#from tvm.contrib import graph_runtime
from tvm.contrib.debugging import debug_runtime as graph_runtime
```

**Output dumping**
The outputs are dumped to a temporary folder in `/tmp/tvmdbg` folder
1. Graph dumping
     The optimized graph build by nnvm in json format is dumped as it is.
2. Tensor dumping
     The output tensors for each node in the graph is saved in numpy format.

The above two modifications will bring up the debug UI during run.

The HOME page of tvmdbg looks like below.
 ![](https://raw.githubusercontent.com/dmlc/web-data/master/tvm/docs/dev/tvmdbg_images/tvm_dbg1.png)

Here user will get the option to run with or without debugging.
Once user perfoms the run, it will take you for listing the nodes in graph.
 ![](https://raw.githubusercontent.com/dmlc/web-data/master/tvm/docs/dev/tvmdbg_images/tvm_dbg2.png)

By clicking at the **Node name** user can see the node details, like its
1. Node information and its attributes
2. Node inputs
3. Node outputs
4. Its computations output values
 ![](https://raw.githubusercontent.com/dmlc/web-data/master/tvm/docs/dev/tvmdbg_images/tvm_dbg3.png)
 ![](https://raw.githubusercontent.com/dmlc/web-data/master/tvm/docs/dev/tvmdbg_images/tvm_dbg4.png)
 ![](https://raw.githubusercontent.com/dmlc/web-data/master/tvm/docs/dev/tvmdbg_images/tvm_dbg5.png)
 ![](https://raw.githubusercontent.com/dmlc/web-data/master/tvm/docs/dev/tvmdbg_images/tvm_dbg6.png)


**Limitations:**
1. Can dump only fused graph, if need to see each and every operation seperately, disable the nnvm optimizations
2. Layer information may be dispersed into multiple operators.

**References**

1. https://github.com/tensorflow/tensorflow
2. https://github.com/tensorflow/tensorboard
3. https://github.com/awslabs/mxboard

