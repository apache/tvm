# NNVM: Open Compiler for AI Frameworks
[![Build Status](http://mode-gpu.cs.washington.edu:8080/buildStatus/icon?job=dmlc/nnvm/master)](http://mode-gpu.cs.washington.edu:8080/job/dmlc/job/nnvm/job/master/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

[Installation](docs/how_to/install.md) |
[Documentation](http://nnvm.tvmlang.org) |
[Tutorials](http://nnvm.tvmlang.org/tutorials/index.html) |
[Release Notes](NEWS.md)

NNVM compiler offers reusable computation graph optimization and compilation for deep learning systems.
It is backed by the [TVM stack](http://tvmlang.org) and provides modules to:

- Represent deep learning workloads from front-end frameworks via a graph IR.
- Optimize computation graphs to improve performance.
- Compile into executable modules and deploy to different hardware backends with minimum dependency.

NNVM is designed to add new frontend, operators and graph optimizations in a decentralized fashion without changing the core interface.
The compiled module can be deployed to server, mobile, embedded devices and browsers with minimum dependency, in languages including c++, python, javascript, java, objective-c. Checkout [our release announcement](http://www.tvmlang.org/2017/10/06/nnvm-compiler-announcement.html)

The following code snippet demonstrates the general workflow of nnvm compiler.

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

License
-------
Licensed under an [Apache-2.0](https://github.com/dmlc/tvm/blob/master/LICENSE) license.


Links
-----
- [TinyFlow](https://github.com/tqchen/tinyflow) on how you can use  NNVM to build a TensorFlow like API.
- [Apache MXNet](http://mxnet.io/) uses NNVM as a backend.
