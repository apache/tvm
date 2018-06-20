# NNVM Compiler Module of TVM Stack

```python
import tvm
from tvm.contrib import graph_runtime, rpc
import nnvm.frontend
import nnvm.compiler

# GET model from frameworks
# change xyz to supported framework name.
graph, params = nnvm.frontend.from_xyz(...)

# OPTIMIZE and COMPILE the graph to get a deployable module
# target can be "opencl", "llvm", "metal" or any target supported by tvm
target = "cuda"
graph, lib, params = nnvm.compiler.build(graph, target, {"data", data_shape}, params=params)

# DEPLOY and run on gpu(0)
module = graph_runtime.create(graph, lib, tvm.gpu(0))
module.set_input(**params)
module.run(data=data_array)
output = tvm.nd.empty(out_shape, ctx=tvm.gpu(0))
module.get_output(0, output)

# DEPLOY to REMOTE mobile/rasp/browser with minimum tvm rpc runtime
# useful for quick experiments on mobile devices
remote = rpc.connect(remote_host, remote_port)
lib.export_library("mylib.so")
remote.upload("mylib.so")
rlib = rpc.load_module("mylib.so")
# run on remote device
rmodule = graph_runtime.create(graph, rlib, remote.gpu(0))
rmodule.set_input(**params)
rmodule.run()
```
