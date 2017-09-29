Deploy Compiled Modules
=======================
NNVM compiled modules are fully embedded in TVM runtime as long as ```GRAPH_RUNTIME``` option
is enabled in tvm runtime. Check out the [TVM documentation](http://docs.tvmlang.org/) for
how to deploy TVM runtime to your system.

In a nutshell, we will need three items to deploy a compiled module.
Checkout our tutorials on getting started with NNVM compiler for more details.

- The graph json data which contains the execution graph.
- The tvm module library of compiled functions.
- The parameter blobs for stored parameters.

We can then use TVM's runtime API to deploy the compiled module.
Here is an example in python.

```python
import tvm

# tvm module for compiled functions.
loaded_lib = tvm.module.load("deploy.so")
# json graph
loaded_json = open(temp.relpath("deploy.json")).read()
# parameters in binary
loaded_params = bytearray(open(temp.relpath("deploy.params"), "rb").read())

fcreate = tvm.get_global_func("tvm.graph_runtime.create")
ctx = tvm.gpu(0)
gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]
set_input("x", tvm.nd.array(x_np))
gmodule["load_params"](loaded_params)
run()
out = tvm.nd.empty(shape)
get_output(0, out)
print(out.asnumpy())
```
