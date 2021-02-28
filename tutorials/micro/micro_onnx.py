#!/usr/bin/env python3

# This is a MicroTVM version of `mnist.py`.

import datetime
import io
import os
import sys

import onnx
import tvm
import tvm.micro
from tvm import autotvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
from tvm.micro.contrib import zephyr
from PIL import Image
import numpy as np

MODEL_FILE = "models/mnist-8.onnx"
MODEL_SHAPE = (1, 1, 28, 28)
DIGIT_2_IMAGE = "data/digit-2.jpg"
DIGIT_9_IMAGE = "data/digit-9.jpg"
INPUT_TENSOR_NAME = "Input3"


def relay_build(mod, params):
    TVM_OPT_LEVEL = 3
    target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=TVM_OPT_LEVEL):
        lib = relay.build(mod, target, params=params)

    # Instantiate the Graph runtime.
    gmodule = runtime.GraphModule(lib["default"](tvm.cpu()))
    print("Built module is:")
    print(gmodule)
    return gmodule


def run_inference(gmodule, sess, perf_eval=True):

    # Load test images.
    digit_2 = Image.open(DIGIT_2_IMAGE).resize((28, 28))
    digit_9 = Image.open(DIGIT_9_IMAGE).resize((28, 28))
    digit_2 = np.asarray(digit_2).astype("float32")
    digit_9 = np.asarray(digit_9).astype("float32")
    digit_2 = np.expand_dims(digit_2, axis=0)
    digit_9 = np.expand_dims(digit_9, axis=0)

    gmodule.set_input(INPUT_TENSOR_NAME, tvm.nd.array(digit_2))
    gmodule.run()
    sess.context.sync()
    output = gmodule.get_output(0).asnumpy()
    print(f"Top result for digit-2 is: {np.argmax(output)}")

    gmodule.set_input(INPUT_TENSOR_NAME, tvm.nd.array(digit_9))
    gmodule.run()
    output = gmodule.get_output(0).asnumpy()
    print(f"Top result for digit-9 is: {np.argmax(output)}")

    if perf_eval:
        print("Performance eval...")
        ctx = tvm.cpu()
        ftimer = gmodule.module.time_evaluator("run", ctx, number=25, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


UTVM_MODEL = "nrf5340dk"
UTVM_ZEPHYR_BOARD = "nrf5340dk_nrf5340_cpuapp"
#UTVM_MODEL = "host"
#UTVM_ZEPHYR_BOARD = "qemu_x86"
UTVM_ZEPHYR_RUNTIME_DIR = "../../apps/microtvm/zephyr/demo_runtime/"
UTVM_WEST_CMD = "west"


onnx_model = onnx.load(MODEL_FILE)

print(f"Loaded ONNX model: {MODEL_FILE}")
print(type(onnx_model))

print(f"IR version: {onnx_model.ir_version}")
print(f"Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
print(f"Domain: {onnx_model.domain}")
print(f"Version: {onnx_model.model_version}")
print(f"Doc string: {onnx_model.doc_string}")
print(f"Graph name: {onnx_model.graph.name}")
print(f"Graph doc string: {onnx_model.graph.doc_string}")

print("Graph:")
print(onnx.helper.printable_graph(onnx_model.graph))

# Convert to Relay.
relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=MODEL_SHAPE, freeze_params=True)
relay_mod = relay.transform.DynamicToStatic()(relay_mod)

# Do the build.
# We add -link-params here so the model parameters are included in the compiled build.
target = tvm.target.target.micro(UTVM_MODEL, options=["-link-params=1"])
TVM_OPT_LEVEL = 3
with tvm.transform.PassContext(opt_level=TVM_OPT_LEVEL, config={"tir.disable_vectorize": True}):
    lowered = relay.build(relay_mod, target, params=params)
    graph_json_str = lowered.get_json()

# gmodule = runtime.GraphModule(lib["default"](target))

# Create a uTVM Workspace.
workspace_root = os.path.abspath(
    f'workspace/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
)
print(f"Using workspace: {workspace_root}")
workspace_parent = os.path.dirname(workspace_root)
if not os.path.exists(workspace_parent):
    os.makedirs(workspace_parent)
workspace = tvm.micro.Workspace(debug=True, root=workspace_root)

# Create Zephyr compiler.
compiler = zephyr.ZephyrCompiler(
    project_dir=UTVM_ZEPHYR_RUNTIME_DIR,
    board=UTVM_ZEPHYR_BOARD,
    zephyr_toolchain_variant="zephyr",
    west_cmd=UTVM_WEST_CMD,
)

# Do the actual build.
opts = tvm.micro.default_options(f"{UTVM_ZEPHYR_RUNTIME_DIR}/crt")
opts["bin_opts"]["ccflags"] = ["-std=gnu++14"]
opts["lib_opts"]["ccflags"] = ["-std=gnu++14"]

micro_bin = tvm.micro.build_static_runtime(
    workspace,
    compiler,
    lowered.lib,
    lib_opts=opts["lib_opts"],
    bin_opts=opts["bin_opts"],
)

build_tarfile = "./build.tar"
if os.path.exists(build_tarfile):
    os.unlink(build_tarfile)
micro_bin.archive(build_tarfile, metadata_only=True)

DEBUG = False
if DEBUG:
    debug_rpc_session = tvm.rpc.connect("127.0.0.1", 9090)
    flasher = compiler.flasher(debug_rpc_session=debug_rpc_session, flash_args=["--softreset"])
else:
    flasher = compiler.flasher(flash_args=["--softreset"])

with tvm.micro.Session(binary=micro_bin, flasher=flasher) as sess:
    mod = tvm.micro.create_local_graph_runtime(graph_json_str, sess.get_system_lib(), sess.context)
    print(f"Created runtime: {mod}")
    print(f"sess.context.sync()...")

    # Populate model parameters.
    #for k, v in lowered.params.items():
    #    num_bytes = np.prod(v.shape) * 4
    #    print(f"MDW: Setting parameter: {k} with size {v.shape}, {num_bytes} bytes")
    #    mod.set_input(k, v)

    sess.context.sync()  # Ensure all args have been pushed to the device.

    print(f"mod is: {mod}")
    print(f"mod.run is: {mod._run}")
    print(f"Running inference...")

    run_inference(mod, sess, perf_eval=False)
