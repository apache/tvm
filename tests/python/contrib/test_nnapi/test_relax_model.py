from tvm.relax.backend.contrib.nnapi import partition_for_nnapi

import tvm
from tvm.relax.transform.transform import RunCodegen
import tvm.script
import tvm.script.relax as R
import tvm.script.tir as T
from tvm import rpc, relay
from tvm.contrib import utils, ndk
from tvm.relax.frontend.onnx import from_onnx

from tvm.relax.testing import relay_translator
from typing import Dict
from torchvision import transforms, datasets
from tvm.contrib.download import download_testdata

from typing import List, Tuple
import pytest
import numpy as np
import os

import onnx

proxy_host = os.environ.get("TVM_ANDROID_RPC_PROXY_HOST", "127.0.0.1")
proxy_port = os.environ.get("TVM_ANDROID_RPC_PROXY_PORT", 9190)
destination = os.environ.get("TVM_ANDROID_RPC_DESTINATION", "")
key = "android"

def reshape_matmul(mod: tvm.IRModule):
    from typing import Dict
    from tvm.relax.dpl import rewrite_call, DFPattern
    from tvm.relax import Expr
    from tvm.relax.dpl.pattern import (
        is_op,
        wildcard,
    )
    input0 = wildcard()
    input1 = wildcard()
    pattern = is_op("relax.matmul")(input0, input1)
    def _rewriter(expr: Expr, matches: Dict[DFPattern, Expr]):
        i0 = matches[input0]
        i1 = matches[input1]
        if len(i0.struct_info.shape) == 2 and len(i1.struct_info.shape) == 2:
            i0_shape = [1] + [*i0.struct_info.shape.values]
            i1_shape = [1] + [*i1.struct_info.shape.values]
            oshape = matches[pattern].struct_info.shape
            return R.reshape(R.matmul(R.reshape(i0, i0_shape), R.reshape(i1, i1_shape)), oshape)
        return expr
    mod["main"] = rewrite_call(pattern, _rewriter, mod["main"])
    return mod

def decompose_clip(mod: tvm.IRModule) -> tvm.IRModule:
    from typing import Dict
    from tvm.relax.dpl import rewrite_call, DFPattern
    from tvm.relax import Expr
    from tvm.relax.dpl.pattern import (
        is_op,
        wildcard,
    )

    input_pattern = wildcard()
    min_pattern = wildcard()
    max_pattern = wildcard()
    pattern = is_op("relax.clip")(input_pattern, min_pattern, max_pattern)
    def _rewriter(expr: Expr, matches: Dict[DFPattern, Expr]) -> Expr:
        dtype = matches[input_pattern].struct_info.dtype
        # import pdb; pdb.set_trace()
        return R.minimum(R.maximum(matches[input_pattern], R.const(np.array(matches[min_pattern].value.value).astype(dtype), dtype)), R.const(np.array(matches[max_pattern].value.value).astype(dtype), dtype))
    mod["main"] = rewrite_call(pattern, _rewriter, mod["main"])
    return mod

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


def build_for_android(mod: tvm.IRModule, enable_nnapi: bool) -> tvm.relax.Executable:
    if enable_nnapi:
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = reshape_matmul(mod)
        mod = decompose_clip(mod)
        mod = partition_for_nnapi(mod)
        
        mod = tvm.relax.transform.RunCodegen()(mod)
    ex = tvm.relax.build(mod, target='llvm -mtriple=aarch64-linux-android')
    return ex

def execute_on_nnapi(mod: tvm.IRModule, inputs: List[tvm.nd.NDArray], enable_nnapi: bool, devices_spec: str) -> np.ndarray:
    tmp = utils.tempdir()
    so_name = "test_mod.so"

    # Build and export library for Android.
    ex = build_for_android(mod, enable_nnapi=enable_nnapi)
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

def execute_on_host(mod: tvm.IRModule, inputs: List):
    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.relax.build(mod, target="llvm")
    dev = tvm.cpu(0)
    vm = tvm.relax.VirtualMachine(ex, device=dev)
    output = vm["main"](*inputs)

    return output.numpy()

def test_vgg11():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg11_Opset18_timm/vgg11_Opset18.onnx"
        model_path = download_testdata(model_url, "vgg11.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod = from_onnx(onnx_model, shape_dict)
        print(mod)
        return mod
    
    mod = create_model()
    verify(
        mod,
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
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
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
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
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
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
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
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
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
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
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
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
        inputs={
            "x": img
        },
        decimal=2
    )

def test_vgg19():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg19_Opset18_timm/vgg19_Opset18.onnx"
        model_path = download_testdata(model_url, "vgg19.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
        inputs={
            "x": img
        },
        decimal=2
    )

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
        mod = from_onnx(onnx_model, shape_dict)
        
        return mod

    
    mod = create_model()
    verify(mod, inputs={"import/Placeholder:0": image}, decimal=1)

def test_mobilenet_v3():
    img = input_img()
    def create_model():
        
        model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/mobilenetv3_large_100_miil_Opset17_timm/mobilenetv3_large_100_miil_Opset17.onnx"
        model_path = download_testdata(model_url, "mobilenet_v3.onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": img.shape}
        mod = from_onnx(onnx_model, shape_dict)
        return mod
    
    mod = create_model()
    verify(
        mod,
        inputs={
            "x": img
        },
        decimal=2
    )

def verify(mod: tvm.IRModule ,inputs: Dict[str, np.ndarray] = {}, decimal: int = 7):
    inputs_tvm: List[tvm.nd.NDArray] = [tvm.nd.array(v) for k, v in inputs.items()]
    host_output = execute_on_host(mod, inputs_tvm)
    nnapi_output = execute_on_nnapi(mod, inputs_tvm, enable_nnapi=True, devices_spec="")
    np.testing.assert_almost_equal(nnapi_output, host_output, decimal=2)
    



