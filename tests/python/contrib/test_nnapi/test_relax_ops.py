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

from tvm.relax.backend.contrib.nnapi import partition_for_nnapi

import tvm
import tvm.script
import tvm.script.relax as R
import tvm.script.tir as T
from tvm import rpc
from tvm.contrib import utils, ndk

from typing import List
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
        R.floor,
        R.nn.relu,
        R.sigmoid,
        R.nn.softmax,
        R.tanh,
        R.abs,
        R.exp,
        R.log,
        R.negative,
        R.sqrt,
        R.rsqrt,
    ],
)
def test_unary(op):
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor(("n", 2, 8, 5), "float32")
            ) -> R.Tensor(("n", 2, 8, 5), "float32"):
                with R.dataflow():
                    t0 = op(i0)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[np.random.uniform(size=(10, 2, 8, 5)).astype("float32")],
        decimal=5,
    )


@pytest.mark.parametrize(
    "op",
    [
        R.add,
        R.multiply,
        R.subtract,
        R.equal,
        R.greater,
        R.greater_equal,
        R.less,
        R.less_equal,
        R.not_equal,
        R.maximum,
        R.minimum,
        R.power,
    ],
)
def test_elementwise_binary(op):
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor(("n", 2, 8, 5), "float32"),
                i1: R.Tensor(("n", 2, 8, 5), "float32"),
            ) -> R.Tensor(("n", 2, 8, 5), "float32"):
                with R.dataflow():
                    t0 = op(i0, i1)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[
            np.random.uniform(size=(10, 2, 8, 5)).astype("float32"),
            np.random.uniform(size=(10, 2, 8, 5)).astype("float32"),
        ],
        decimal=5,
    )


def test_divide():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor(("n", 2, 8, 5), "float32"),
                i1: R.Tensor(("n", 2, 8, 5), "float32"),
            ) -> R.Tensor(("n", 2, 8, 5), "float32"):
                with R.dataflow():
                    t0 = R.divide(i0, i1)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[
            np.random.uniform(size=(10, 2, 8, 5)).astype("float32"),
            np.random.uniform(size=(10, 2, 8, 5)).astype("float32")
            + np.ones((10, 2, 8, 5), "float32"),
        ],
        decimal=5,
    )


def test_matmul():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((5, 3, 4), "float32"),
                i1: R.Tensor((5, 4, 8), "float32"),
            ) -> R.Tensor((5, 3, 8), "float32"):
                with R.dataflow():
                    t0 = R.matmul(i0, i1)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[
            np.random.random(size=(5, 3, 4)).astype("float32"),
            np.random.random(size=(5, 4, 8)).astype("float32"),
        ],
    )


def test_permute_dims():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((5, 4, 8), "float32"),
            ) -> R.Tensor((8, 5, 4), "float32"):
                with R.dataflow():
                    t0 = R.permute_dims(i0, axes=[2, 0, 1])
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[
            np.random.random(size=(5, 4, 8)).astype("float32"),
        ],
    )


def test_astype():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((8, 10, 15), "float32"),
            ) -> R.Tensor((8, 10, 15), "float16"):
                with R.dataflow():
                    t0: R.Tensor((8, 10, 15), "float16") = R.astype(i0, dtype="float16")
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[
            tvm.nd.array(np.random.uniform(size=(8, 10, 15)).astype("float32")),
        ],
    )


def test_mean():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor(("n", 10, 15), "float32"),
            ) -> R.Tensor(("n", 10, 1), "float32"):
                n = T.int64()
                with R.dataflow():
                    t0: R.Tensor((n, 10, 15), "float32") = R.mean(i0, axis=[-1], keepdims=True)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        mod,
        inputs=[
            tvm.nd.array(np.random.uniform(size=(8, 10, 15)).astype("float32")),
        ],
    )


def test_conv2d():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 3, 224, 224), "float32"),
                i1: R.Tensor((64, 3, 3, 3), "float32"),
                i2: R.Tensor((1, 64, 1, 1), "float32"),
            ):
                with R.dataflow():
                    t0 = R.nn.conv2d(i0, i1, strides=(1, 1), padding=(1, 1))
                    t0 = R.add(i2, t0)
                    R.output(t0)
                return t0
        
        return Module
    
    mod = create_model()
    verify(
        mod,
        inputs=[
            np.random.random(size=(1, 3, 224, 224)).astype("float32"),
            np.random.random(size=(64, 3, 3, 3)).astype("float32"),
            np.random.random(size=(1, 64, 1, 1)).astype("float32"),
        ],
        decimal=2
    )

def test_max_pool2d():
    def create_model() -> tvm.IRModule:
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 1, 28, 28), "float32"),
            ):
                with R.dataflow():
                    t0 = R.nn.max_pool2d(i0, pool_size=(1,1), strides=(1, 1), padding=(0, 0))
                    R.output(t0)
                return t0
        
        return Module
    
    mod = create_model()
    verify(
        mod,
        inputs=[
            np.random.random(size=(1, 1, 28, 28)).astype("float32"),
        ],
    )

def build_for_nnapi(mod: tvm.IRModule) -> tvm.relax.Executable:
    mod = partition_for_nnapi(mod, feature_level=7)
    mod = tvm.relax.transform.RunCodegen()(mod)
    ex = tvm.relax.build(mod, target="llvm -mtriple=aarch64-linux-android")
    return ex


def build_for_host(mod: tvm.IRModule) -> tvm.relax.Executable:
    ex = tvm.relax.build(mod, target="llvm")
    return ex


def execute_on_nnapi(mod: tvm.IRModule, inputs: List[tvm.nd.NDArray]) -> np.ndarray:
    tmp = utils.tempdir()
    so_name = "test_mod.so"

    # Build and export library for Android.
    ex = build_for_nnapi(mod)
    so_path = tmp / so_name
    ex.export_library(str(so_path), fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])

    # Upload the shared library to the remote.
    tracker = rpc.connect_tracker(proxy_host, proxy_port)
    remote = tracker.request(key, priority=0)
    dev = remote.cpu(0)
    remote.upload(so_path)

    try:
        # Execute the model on the remote.
        remote_ex = remote.load_module(so_name)
        vm = tvm.relax.VirtualMachine(remote_ex, device=dev)
        inputs = [x.copyto(dev) for x in inputs]
        vm.set_input("main", *inputs)
        vm.invoke_stateful("main")
        output = vm.get_outputs("main")
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


def execute_on_host(mod: tvm.IRModule, inputs: List) -> np.ndarray:
    ex = build_for_host(mod)
    dev = tvm.cpu(0)
    vm = tvm.relax.VirtualMachine(ex, device=dev)
    output = vm["main"](*inputs)

    return output.numpy()


def verify(mod: tvm.IRModule, inputs: List[np.ndarray] = [], decimal: int = 7):
    inputs_tvm: List[tvm.nd.NDArray] = [tvm.nd.array(x) for x in inputs]
    host_output = execute_on_host(mod, inputs_tvm)
    nnapi_output = execute_on_nnapi(mod, inputs_tvm)
    np.testing.assert_almost_equal(nnapi_output, host_output, decimal=decimal)
