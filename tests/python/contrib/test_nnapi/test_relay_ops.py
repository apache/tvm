# NNAPI tests using RPC.
#
# To run the tests with pytest, set the environment variable RUN_NNAPI_TEST to 1.
# Alternatively, run this file directly with python.
#
# Requirements to run the tests:
# - The environment variable TVM_NDK_CC set to a C compiler from NDK.
# - An RPC tracker setup at 127.0.0.1:9190 by default.
#   Use TVM_ANDROID_RPC_PROXY_{HOST,PORT} environment variables to override.
# - An RPC server running on Android with the key "android" is required, and the RPC server must be
#   built with NNAPI runtime support.

from tvm.contrib import graph_executor
from tvm.contrib.graph_executor import GraphModule
from tvm.relay.build_module import GraphExecutor
from tvm.relay.op.contrib.nnapi import partition_for_nnapi

import tvm
from tvm import rpc, relay
from tvm.contrib import utils, ndk
from tvm.relay.backend.executor_factory import GraphExecutorFactoryModule

from typing import Dict
import pytest
import numpy as np
import os


proxy_host = os.environ.get("TVM_ANDROID_RPC_PROXY_HOST", "127.0.0.1")
proxy_port = os.environ.get("TVM_ANDROID_RPC_PROXY_PORT", 9190)
destination = os.environ.get("TVM_ANDROID_RPC_DESTINATION", "")
key = "android"


def skip_nnapi_tests() -> bool:
    return os.getenv("RUN_NNAPI_TESTS") != "1"


if skip_nnapi_tests():
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    "op",
    [
        relay.floor,
        relay.nn.relu,
        relay.sigmoid,
        relay.tanh,
        relay.abs,
        relay.exp,
        relay.log,
        relay.negative,
    ],
)
def test_unary(op):
    def create_model() -> tvm.IRModule:
        input0 = relay.var("input0", relay.TensorType((8, 5, 3), "float32"))
        output = op(input0)
        func = relay.Function([input0], output)
        return tvm.IRModule.from_expr(func)

    mod = create_model()
    verify(
        mod,
        inputs={
            "input0": np.random.uniform(size=(8, 5, 3)).astype("float32"),
        },
        decimal=5,
    )


@pytest.mark.parametrize(
    "op",
    [
        relay.add,
        relay.multiply,
        relay.subtract,
        relay.equal,
        relay.greater,
        relay.greater_equal,
        relay.less,
        relay.less_equal,
        relay.not_equal,
    ],
)
def test_elementwise_binary(op):
    def create_model() -> tvm.IRModule:
        input0 = relay.var("input0", relay.TensorType((8, 5, 3), "float32"))
        input1 = relay.var("input1", relay.TensorType((8, 5, 3), "float32"))
        output = op(input0, input1)
        func = relay.Function([input0, input1], output)
        return tvm.IRModule.from_expr(func)

    mod = create_model()
    verify(
        mod,
        inputs={
            "input0": np.random.uniform(size=(8, 5, 3)).astype("float32"),
            "input1": np.random.uniform(size=(8, 5, 3)).astype("float32"),
        },
    )


def test_conv2d_with_bias():
    def create_model() -> tvm.IRModule:
        stride = (2, 2)
        padding = (1, 2)
        input0 = relay.var("input0", relay.TensorType((1, 2, 10, 10), "float32"))
        input1 = relay.var("input1", relay.TensorType((1, 2, 3, 3), "float32"))
        input2 = relay.var("input2", relay.TensorType((1,), "float32"))
        middle = relay.nn.conv2d(input0, input1, stride, padding)
        output = relay.nn.bias_add(middle, input2)
        func = relay.Function([input0, input1, input2], output)
        return tvm.IRModule.from_expr(func)

    mod = create_model()
    verify(
        mod,
        inputs={
            "input0": np.random.uniform(size=(1, 2, 10, 10)).astype("float32"),
            "input1": np.random.uniform(size=(1, 2, 3, 3)).astype("float32"),
            "input2": np.random.uniform(size=(1,)).astype("float32"),
        },
        decimal=5,
    )


@pytest.mark.parametrize(
    "data,kernel",
    [
        ((1, 1, 28, 28), (1, 1, 3, 3)),
        ((5, 1, 28, 28), (1, 1, 4, 4)),
    ],
)
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize(
    "padding",
    [
        (0, 0),
        (1, 1),
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_conv2d(data, kernel, stride, padding, dtype):
    def create_model() -> tvm.IRModule:
        input0 = relay.var("input0", relay.TensorType(data, dtype))
        input1 = relay.var("input1", relay.TensorType(kernel, dtype))
        output = relay.nn.conv2d(input0, input1, stride, padding)

        func = relay.Function([input0, input1], output)
        return tvm.IRModule.from_expr(func)

    mod = create_model()

    if dtype == "float16":
        decimal = 2
    else:
        decimal = 5
    verify(
        mod,
        inputs={
            "input0": np.random.uniform(size=data).astype(dtype),
            "input1": np.random.uniform(size=kernel).astype(dtype),
        },
        decimal=decimal,
    )


def test_dense_with_bias():
    def create_model() -> tvm.IRModule:
        input0 = relay.var("input0", relay.TensorType((5, 3), "float32"))
        input1 = relay.var("input1", relay.TensorType((4, 3), "float32"))
        input2 = relay.var("input2", relay.TensorType((4,), "float32"))
        middle = relay.nn.dense(input0, input1)
        output = relay.nn.bias_add(middle, input2)
        func = relay.Function([input0, input1, input2], output)
        return tvm.IRModule.from_expr(func)

    mod = create_model()
    verify(
        mod,
        inputs={
            "input0": np.random.uniform(size=(5, 3)).astype("float32"),
            "input1": np.random.uniform(size=(4, 3)).astype("float32"),
            "input2": np.random.uniform(size=(4,)).astype("float32"),
        },
        decimal=5,
    )

def test_max_pool2d():
    def create_model() -> tvm.IRModule:
        input0 = relay.var("input0", relay.TensorType((1, 1, 16, 16),dtype="float16"))
        max_pool = relay.nn.max_pool2d(input0)
        func = relay.Function([input0], max_pool)
        return tvm.IRModule.from_expr(func)

    mod = create_model()
    verify(
        mod,
        inputs={
            "input0": np.random.uniform(size=(1, 1, 16, 16)).astype("float16"),
        },
        decimal=3,
    )

def build_for_nnapi(mod: tvm.IRModule) -> GraphExecutorFactoryModule:
    mod = partition_for_nnapi(mod)
    lib = relay.build(mod, target="llvm -mtriple=aarch64-linux-android")
    return lib


def build_for_host(mod: tvm.IRModule) -> GraphExecutorFactoryModule:
    lib = relay.build(mod, target="llvm")
    return lib


def execute_on_nnapi(mod: tvm.IRModule, inputs: Dict[str, tvm.nd.NDArray]) -> np.ndarray:
    tmp = utils.tempdir()
    so_name = "test_mod.so"

    # Build and export library for Android.
    lib = build_for_nnapi(mod)
    so_path = tmp / so_name
    lib.export_library(
        str(so_path), fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"]
    )

    # Upload the shared library to the remote.
    tracker = rpc.connect_tracker(proxy_host, proxy_port)
    remote = tracker.request(key, priority=0)
    dev = remote.cpu(0)
    remote.upload(so_path)

    try:
        # Execute the model on the remote.
        remote_lib = remote.load_module(so_name)
        graph_mod = graph_executor.GraphModule(remote_lib["default"](dev))
        for name, value in inputs.items():
            graph_mod.set_input(name, value)
        graph_mod.run()
        output = graph_mod.get_output(0)
        output = output.numpy()
    except Exception as e:
        # Re-raise all exceptions
        raise e
    finally:
        # Manually close the connection.
        # See https://discuss.tvm.apache.org/t/trouble-with-rpc-session/14008/.
        #
        # TODO: Remove if it does not happen on Python 3.11.
        remote._sess.get_function("CloseRPCConnection")()
        tracker.close()

    return output


def execute_on_host(mod: tvm.IRModule, inputs: Dict[str, tvm.nd.NDArray]) -> np.ndarray:
    lib = build_for_host(mod)
    dev = tvm.cpu(0)
    graph_mod = graph_executor.GraphModule(lib["default"](dev))
    for name, value in inputs.items():
        graph_mod.set_input(name, value)
    graph_mod.run()
    output = graph_mod.get_output(0)

    return output.numpy()


def verify(mod: tvm.IRModule, inputs: Dict[str, np.ndarray] = {}, decimal: int = 7):
    inputs_tvm: Dict[str, tvm.nd.NDArray] = {k: tvm.nd.array(v) for k, v in inputs.items()}
    host_output = execute_on_host(mod, inputs_tvm)
    nnapi_output = execute_on_nnapi(mod, inputs_tvm)
    np.testing.assert_almost_equal(nnapi_output, host_output, decimal=decimal)
