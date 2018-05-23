# some standard imports
import nnvm
import tvm
from nnvm.compiler import graph_attr
import vta
import vta.testing
import os
import numpy as np
from PIL import Image
import pickle
import json
import logging
import wget
from tvm.contrib import graph_runtime, rpc, util

bfactor = 1
cfactor = 16
verbose = False
# only run fpga component, mark non-conv ops as nop
debug_fpga_only = False

# Obtain model and hardware files (they're too large to check-in)
url = "https://homes.cs.washington.edu/~moreau/media/vta/"
TEST_FILE = 'cat.jpg'
CATEG_FILE = 'synset.txt'
RESNET_GRAPH_FILE = 'quantize_graph.json'
RESNET_PARAMS_FILE = 'quantize_params.pkl'
for file in [TEST_FILE, CATEG_FILE, RESNET_GRAPH_FILE, RESNET_PARAMS_FILE]:
    if not os.path.isfile(file):
        print ("Downloading {}".format(file))
        wget.download(url+file)

if verbose:
    logging.basicConfig(level=logging.DEBUG)

# Change to -device=vtacpu to run cpu only inference.
target = tvm.target.create("llvm -device=vta")
target_host = "llvm -mtriple=armv7-none-linux-gnueabihf -mcpu=cortex-a9 -mattr=+neon"

if vta.get_env().TARGET == "sim":
    target_host = "llvm"

synset = eval(open(os.path.join(CATEG_FILE)).read())
image = Image.open(os.path.join(TEST_FILE)).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def mark_nop(graph, conv_layer=-1, skip_conv_layer=()):
    """Helper function to mark certain op as nop

    Useful to debug performance issues.
    """
    jgraph = json.loads(graph.json())
    counter = 0
    for nid, node in enumerate(jgraph["nodes"]):
        op_name = node["op"]
        if op_name != "tvm_op":
            continue
        attrs = node["attrs"]
        node_name = node["name"]
        func_name = attrs["func_name"]
        if func_name.find("quantized_conv2d") != -1:
            if conv_layer >= 0:
                if counter != conv_layer:
                    attrs["func_name"] = "__nop"
            if counter in skip_conv_layer:
                attrs["func_name"] = "__nop"
            counter += 1
        else:
            if conv_layer >= 0:
                attrs["func_name"] = "__nop"
            attrs["func_name"] = "__nop"
        if attrs["func_name"] != "__nop":
            print("Run function %s"% func_name)
    graph = nnvm.graph.load_json(json.dumps(jgraph))
    return graph

x = transform_image(image)
print('x', x.shape)

######################################################################
# now compile the graph
import nnvm.compiler
np.random.seed(0)
sym = nnvm.graph.load_json(
    open(os.path.join(RESNET_GRAPH_FILE)).read())
params = pickle.load(
    open(os.path.join(RESNET_PARAMS_FILE)))

shape_dict = {"data": x.shape}
dtype_dict = {"data": 'float32'}
shape_dict.update({k: v.shape for k, v in params.items()})
dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

graph = nnvm.graph.create(sym)
graph_attr.set_shape_inputs(sym, shape_dict)
graph_attr.set_dtype_inputs(sym, dtype_dict)
graph = graph.apply("InferShape").apply("InferType")

dtype = "float32"
sym = vta.graph.remove_stochastic(sym)
sym = vta.graph.clean_cast(sym)
sym = vta.graph.clean_conv_fuse(sym)
if target.device_name == "vta":
    sym = vta.graph.pack(sym, shape_dict, bfactor, cfactor)

graph_attr.set_shape_inputs(sym, shape_dict)
sym = sym.apply("InferShape")
graph_attr.set_dtype_inputs(sym, dtype_dict)
sym = sym.apply("InferType")

with nnvm.compiler.build_config(opt_level=3):
    if target.device_name != "vta":
        graph, lib, params = nnvm.compiler.build(
            sym, target_host, shape_dict, dtype_dict,
            params=params)
    else:
        with vta.build_config():
            graph, lib, params = nnvm.compiler.build(
                sym, target, shape_dict, dtype_dict,
                params=params, target_host=target_host)


assert tvm.module.enabled("rpc")
temp = util.tempdir()
lib.save(temp.relpath("graphlib.o"))

if vta.get_env().TARGET == "sim":
    remote = rpc.LocalSession()
    print("local session")
else:
    host = os.environ.get("VTA_PYNQ_RPC_HOST", None)
    assert host
    port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
    port = int(port)
    remote = rpc.connect(host, port)

# Program FPGA, and build runtime if necessary
# Overwrite bitstream with a path to your own if you built it yourself
vta.reconfig_runtime(remote)
vta.program_fpga(remote, bitstream=None)

remote.upload(temp.relpath("graphlib.o"))
lib = remote.load_module("graphlib.o")
ctx = remote.ext_dev(0) if target.device_name == "vta" else remote.cpu(0)

print("Build complete...")

def run_e2e(graph):
    """Running end to end example
    """
    if debug_fpga_only:
        graph = mark_nop(graph, skip_conv_layer=(0,))
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype("float32")))
    m.set_input(**params)
    # execute
    timer = m.module.time_evaluator("run", ctx, number=10)
    tcost = timer()
    # get outputs
    tvm_output = m.get_output(
        0,tvm.nd.empty((1000,), dtype, remote.cpu(0)))
    top1 = np.argmax(tvm_output.asnumpy())
    print('TVM prediction top-1:', top1, synset[top1])
    print("t-cost=%g" % tcost.mean)


def run_layer(old_graph, layer_begin, layer_end):
    """Run a certain layer."""
    for layer_id in range(layer_begin, layer_end):
        print("run resnet[%d]..."% (layer_id))
        graph = mark_nop(old_graph, layer_id)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input('data', tvm.nd.array(x.astype("float32")))
        m.set_input(**params)
        # execute
        timer = m.module.time_evaluator("run", ctx, number=1)
        tcost = timer()
        print("resnet[%d]: %g\n"% (layer_id, tcost.mean))

run_e2e(graph)
