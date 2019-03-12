"""
Get Started with NNVM
=====================
**Author**: `Tianqi Chen <https://tqchen.github.io/>`_

This article is an introductory tutorial to workflow in NNVM.
"""
import nnvm.compiler
import nnvm.symbol as sym

######################################################################
# Declare Computation
# -------------------
# We start by describing our need using computational graph.
# Most deep learning frameworks use computation graph to describe
# their computation. In this example, we directly use
# NNVM's API to construct the computational graph.
#
# .. note::
#
#   In a typical deep learning compilation workflow,
#   we can get the models from :any:`nnvm.frontend`
#
# The following code snippet describes :math:`z = x + \sqrt{y}`
# and creates a nnvm graph from the description.
# We can print out the graph ir to check the graph content.

x = sym.Variable("x")
y = sym.Variable("y")
z = sym.elemwise_add(x, sym.sqrt(y))
compute_graph = nnvm.graph.create(z)
print("-------compute graph-------")
print(compute_graph.ir())

######################################################################
# Compile
# -------
# We can call :any:`nnvm.compiler.build` to compile the graph.
# The build function takes a shape parameter which specifies the
# input shape requirement. Here we only need to pass in shape of ``x``
# and the other one will be inferred automatically by NNVM.
#
# The function returns three values. ``deploy_graph`` contains
# the final compiled graph structure. ``lib`` is a :any:`tvm.module.Module`
# that contains compiled CUDA functions. We do not need the ``params``
# in this case.
shape = (4,)
deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target="cuda", shape={"x": shape}, dtype="float32")

######################################################################
# We can print out the IR of ``deploy_graph`` to understand what just
# happened under the hood. We can find that ``deploy_graph`` only
# contains a single operator ``tvm_op``. This is because NNVM
# automatically fused the operator together into one operator.
#
print("-------deploy graph-------")
print(deploy_graph.ir())

######################################################################
# Let us also peek into content of ``lib``.
# Typically a compiled TVM CUDA module contains a host module(lib)
# and a device module(``lib.imported_modules[0]``) that contains the CUDA code.
# We print out the the generated device code here.
# This is exactly a fused CUDA version of kernel that the graph points to.
#
print("-------deploy library-------")
print(lib.imported_modules[0].get_source())

######################################################################
# Deploy and Run
# --------------
# Now that we have have compiled module, let us run it.
# We can use :any:`graph_runtime <tvm.contrib.graph_runtime.create>`
# in tvm to create a deployable :any:`GraphModule <tvm.contrib.graph_runtime.GraphModule>`.
# We can use the :any:`set_input <tvm.contrib.graph_runtime.GraphModule.set_input>`,
# :any:`run <tvm.contrib.graph_runtime.GraphModule.run>` and
# :any:`get_output <tvm.contrib.graph_runtime.GraphModule.get_output>` function
# to set the input, execute the graph and get the output we need.
#
import tvm
import numpy as np
from tvm.contrib import graph_runtime, util

module = graph_runtime.create(deploy_graph, lib, tvm.gpu(0))
x_np = np.array([1, 2, 3, 4]).astype("float32")
y_np = np.array([4, 4, 4, 4]).astype("float32")
# set input to the graph module
module.set_input(x=x_np, y=y_np)
# run forward computation
module.run()
# get the first output
out = module.get_output(0, out=tvm.nd.empty(shape))
print(out.asnumpy())

######################################################################
# Provide Model Parameters
# ------------------------
# Most deep learning models contains two types of inputs: parameters
# that remains fixed during inference and data input that need to
# change for each inference task. It is helpful to provide these
# information to NNVM. Let us assume that ``y`` is the parameter
# in our example. We can provide the model parameter information
# by the params argument to :any:`nnvm.compiler.build`.
#
deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target="cuda", shape={"x": shape}, params={"y": y_np})

######################################################################
# This time we will need params value returned by :any:`nnvm.compiler.build`.
# NNVM applys  optimization  to pre-compute the intermediate values in
# the graph that can be determined by parameters. In this case
# :math:`\sqrt{y}` can be pre-computed. The pre-computed values
# are returned as new params. We can print out the new compiled library
# to confirm that the fused kernel only now contains add.
#
print("-----optimized params-----")
print(params)
print("-------deploy library-------")
print(lib.imported_modules[0].get_source())

######################################################################
# Save the Deployed Module
# ------------------------
# We can save the ``deploy_graph``, ``lib`` and ``params`` separately
# and load them back later. We can use :any:`tvm.module.Module` to export
# the compiled library. ``deploy_graph`` is saved in json format and ``params``
# is serialized into a bytearray.
#
temp = util.tempdir()
path_lib = temp.relpath("deploy.so")
lib.export_library(path_lib)
with open(temp.relpath("deploy.json"), "w") as fo:
    fo.write(deploy_graph.json())
with open(temp.relpath("deploy.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print(temp.listdir())

######################################################################
# We can load the module back.
loaded_lib = tvm.module.load(path_lib)
loaded_json = open(temp.relpath("deploy.json")).read()
loaded_params = bytearray(open(temp.relpath("deploy.params"), "rb").read())
module = graph_runtime.create(loaded_json, loaded_lib, tvm.gpu(0))
params = nnvm.compiler.load_param_dict(loaded_params)
# directly load from byte array
module.load_params(loaded_params)
module.run(x=x_np)
# get the first output
out = module.get_output(0, out=tvm.nd.empty(shape))
print(out.asnumpy())

######################################################################
# Deploy using Another Language
# -----------------------------
# We use python in this example for demonstration.
# We can also deploy the compiled modules with other languages
# supported by TVM such as  c++, java, javascript.
# The graph module itself is fully embedded in TVM runtime.
#
# The following block demonstrates how we can directly use TVM's
# runtime API to execute the compiled module.
# You can find similar runtime API in TVMRuntime of other languages.
#
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
