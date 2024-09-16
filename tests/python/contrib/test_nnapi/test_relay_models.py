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
from torchvision import transforms, datasets
from tvm.contrib.download import download_testdata

import onnx

proxy_host = os.environ.get("TVM_ANDROID_RPC_PROXY_HOST", "127.0.0.1")
proxy_port = os.environ.get("TVM_ANDROID_RPC_PROXY_PORT", 9190)
destination = os.environ.get("TVM_ANDROID_RPC_DESTINATION", "")
key = "android"


def skip_nnapi_tests() -> bool:
    return os.getenv("RUN_NNAPI_TESTS") != "1"


if skip_nnapi_tests():
    pytest.skip(allow_module_level=True)

def input_img():
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    from PIL import Image

    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)
    return img

def test_lenet():
    image = np.abs(np.random.uniform(size=(1, 1, 28, 28))).astype("float32")

    def create_model():
        model_url = "".join(
            [
                "https://github.com/ONNC/onnc-tutorial/raw/af27b015f65339aa07c40d27ffb32fedee7ea692/models/lenet/lenet.onnx",
            ]
        )
        model_path = download_testdata(model_url, "lenet.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"import/Placeholder:0": image.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        
        return mod, params

    
    mod, params = create_model()
    verify(mod, params=params, inputs={"import/Placeholder:0": image}, decimal=1)


def test_vgg19():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg19_Opset18_timm/vgg19_Opset18.onnx"
        model_path = download_testdata(model_url, "vgg19.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )
    
def test_vgg16():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg16_Opset18_timm/vgg16_Opset18.onnx"
        model_path = download_testdata(model_url, "vgg16.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_vgg11():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg11_Opset18_timm/vgg11_Opset18.onnx"
        model_path = download_testdata(model_url, "vgg11.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_mobilenet_v3():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/mobilenetv3_large_100_miil_Opset17_timm/mobilenetv3_large_100_miil_Opset17.onnx"
        model_path = download_testdata(model_url, "mobilenet_v3.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_squeezenet():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/squeezenet1_1_Opset18_torch_hub/squeezenet1_1_Opset18.onnx"
        model_path = download_testdata(model_url, "squeezenet.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params= create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_resnet18():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/resnet18_Opset18_timm/resnet18_Opset18.onnx"
        model_path = download_testdata(model_url, "resnet18.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_resnet34():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/resnet34_Opset18_timm/resnet34_Opset18.onnx"
        model_path = download_testdata(model_url, "resnet34.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_resnet50():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/resnet50_Opset18_timm/resnet50_Opset18.onnx"
        model_path = download_testdata(model_url, "resnet50.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod,params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_alexnet():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/alexnet_Opset17_torch_hub/alexnet_Opset17.onnx"
        model_path = download_testdata(model_url, "alexnet.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params
    
    mod, params = create_model()
    verify(
        mod,
        params=params,
        inputs={
            "x": img
        },
        decimal=2
    )

def build_for_nnapi(mod: tvm.IRModule, params: Dict[str, tvm.nd.NDArray]) -> GraphExecutorFactoryModule:
    mod = partition_for_nnapi(mod)
    lib = relay.build(mod,params=params, target="llvm -mtriple=aarch64-linux-android")
    return lib

def build_for_host(mod: tvm.IRModule, params: Dict[str, tvm.nd.NDArray]) -> GraphExecutorFactoryModule:
    lib = relay.build(mod, params=params, target="llvm")
    return lib

def execute_on_host(mod: tvm.IRModule, params: Dict[str, tvm.nd.NDArray] ,inputs: Dict[str, tvm.nd.NDArray]) -> np.ndarray:
    lib = build_for_host(mod, params)
    dev = tvm.cpu(0)
    graph_mod = graph_executor.GraphModule(lib["default"](dev))
    for name, value in inputs.items():
        graph_mod.set_input(name, value)
    graph_mod.run()
    output = graph_mod.get_output(0)

    return output.numpy()

def execute_on_nnapi(mod: tvm.IRModule, params: Dict[str, tvm.nd.NDArray] ,inputs: Dict[str, tvm.nd.NDArray]) -> np.ndarray:
    tmp = utils.tempdir()
    so_name = "test_mod.so"

    # Build and export library for Android.
    lib = build_for_nnapi(mod, params)
    so_path = tmp / so_name
    lib.export_library(
        str(so_path), fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"]
    )

    # Upload the shared library to the remote.
    tracker = rpc.connect_tracker(proxy_host, proxy_port)
    remote = tracker.request(key, priority=0, session_timeout=6000000)
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

def verify(mod: tvm.IRModule, params: Dict[str, tvm.nd.NDArray] ,inputs: Dict[str, np.ndarray] = {}, decimal: int = 7):
    inputs_tvm: Dict[str, tvm.nd.NDArray] = {k: tvm.nd.array(v) for k, v in inputs.items()}
    host_output = execute_on_host(mod,params ,inputs_tvm)
    nnapi_output = execute_on_nnapi(mod,params ,inputs_tvm)
    np.testing.assert_almost_equal(nnapi_output, host_output, decimal=decimal)
