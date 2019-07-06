# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os

from nose.tools import nottest
import numpy as np
import tvm
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.relay.testing import resnet

# Use the host emulated micro device.
DEVICE_TYPE = "host"
TOOLCHAIN_PREFIX = ""

def create_micro_mod(c_mod, toolchain_prefix):
    """Produces a micro module from a given module.

    Parameters
    ----------
    c_mod : tvm.module.Module
        module with "c" as its target backend

    toolchain_prefix : str
        toolchain prefix to be used (see `tvm.micro.Session` docs)

    Return
    ------
    micro_mod : tvm.module.Module
        micro module for the target device
    """
    temp_dir = util.tempdir()
    # Save module source to temp file.
    lib_src_path = temp_dir.relpath("dev_lib.c")
    # mod_src = c_mod.get_source()
    hardcoded_resnet_path = os.path.join(os.path.dirname(__file__), "resnet_18.c")
    with open(hardcoded_resnet_path, "r") as f:
        mod_src = f.read()
    with open(lib_src_path, "w") as f:
        f.write(mod_src)
    # Compile to object file.
    lib_obj_path = micro.create_micro_lib(lib_src_path, toolchain_prefix)
    micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
    return micro_mod


def relay_micro_build(func, toolchain_prefix, params=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device
    """
    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(func, target="c", params=params)
    micro_mod = create_micro_mod(c_mod, TOOLCHAIN_PREFIX)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


# TODO(weberlo): Add example program to test scalar double/int TVMValue serialization.

def test_alloc():
    """Test tensor allocation on the device."""
    shape = (1024,)
    dtype = "float32"
    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        ctx = tvm.micro_dev(0)
        np_tensor = np.random.uniform(size=shape).astype(dtype)
        micro_tensor = tvm.nd.array(np_tensor, ctx)
        tvm.testing.assert_allclose(np_tensor, micro_tensor.asnumpy())


def test_add():
    """Test a module which performs addition."""
    shape = (1024,)
    dtype = "float32"

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd"
    c_mod = tvm.build(s, [A, B, C], target="c", name=func_name)

    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        micro_mod = create_micro_mod(c_mod, TOOLCHAIN_PREFIX)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        micro_func(a, b, c)

        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_workspace_add():
    """Test a module which uses a workspace to compute an intermediate value."""
    shape = (1024,)
    dtype = "float32"

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd_two_workspace"
    c_mod = tvm.build(s, [A, C], target="c", name=func_name)

    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        micro_mod = create_micro_mod(c_mod, TOOLCHAIN_PREFIX)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        micro_func(a, c)

        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + 2.0)


def test_graph_runtime():
    """Test a program which uses the graph runtime."""
    shape = (1024,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(1.0))
    func = relay.Function([x], z)

    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        mod = relay_micro_build(func, TOOLCHAIN_PREFIX)

        x_in = np.random.uniform(size=shape[0]).astype(dtype)
        mod.run(x=x_in)
        result = mod.get_output(0).asnumpy()

        tvm.testing.assert_allclose(
                result, x_in * x_in + 1.0)


def test_multiple_modules():
    """Test loading multiple modules on the device simultaneously."""
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_func = relay.Function([x], ret)
    # Construct Relay subtract program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.subtract(x, relay.const(1.0))
    sub_func = relay.Function([x], ret)

    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        add_mod = relay_micro_build(add_func, TOOLCHAIN_PREFIX)
        sub_mod = relay_micro_build(sub_func, TOOLCHAIN_PREFIX)

        x_in = np.random.uniform(size=shape[0]).astype(dtype)
        add_mod.run(x=x_in)
        add_result = add_mod.get_output(0).asnumpy()
        sub_mod.run(x=x_in)
        sub_result = sub_mod.get_output(0).asnumpy()

        tvm.testing.assert_allclose(
                add_result, x_in + 1.0)
        tvm.testing.assert_allclose(
                sub_result, x_in - 1.0)


def test_interleave_sessions():
    """Test closing and reopening sessions."""
    shape = (1024,)
    dtype = "float32"

    sess_a = micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX)
    sess_b = micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX)
    with sess_a:
        ctx = tvm.micro_dev(0)
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, ctx)
    with sess_b:
        ctx = tvm.micro_dev(0)
        np_tensor_b = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_b = tvm.nd.array(np_tensor_b, ctx)
    with sess_a:
        tvm.testing.assert_allclose(np_tensor_a, micro_tensor_a.asnumpy())
    with sess_b:
        tvm.testing.assert_allclose(np_tensor_b, micro_tensor_b.asnumpy())


def test_resnet_random():
    """Test ResNet18 inference with random weights and inputs."""
    resnet_func, params = resnet.get_workload(num_classes=10,
                                              num_layers=18,
                                              image_shape=(3, 32, 32))
    # Remove the final softmax layer, because uTVM does not currently support it.
    resnet_func_no_sm = relay.Function(resnet_func.params,
                                       resnet_func.body.args[0],
                                       resnet_func.ret_type)

    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        # TODO(weberlo): Use `resnet_func` once we have libc support.
        mod = relay_micro_build(resnet_func_no_sm, TOOLCHAIN_PREFIX, params=params)
        # Generate random input.
        data = np.random.uniform(size=mod.get_input(0).shape)
        mod.run(data=data)
        result = mod.get_output(0).asnumpy()
        # We gave a random input, so all we want is a result with some nonzero
        # entries.
        assert result.sum() != 0.0


# TODO(weberlo): Enable this test or move the code somewhere else.
@nottest
def test_resnet_pretrained():
    """Test classification with a pretrained ResNet18 model."""
    import mxnet as mx
    from mxnet.gluon.model_zoo.vision import get_model
    from mxnet.gluon.utils import download
    from PIL import Image

    # TODO(weberlo) there's a significant amount of overlap between here and
    # `tutorials/frontend/from_mxnet.py`.  Should refactor.
    dtype = "float32"

    # Fetch a mapping from class IDs to human-readable labels.
    synset_url = "".join(["https://gist.githubusercontent.com/zhreshold/",
                          "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                          "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                          "imagenet1000_clsid_to_human.txt"])
    synset_name = "synset.txt"
    download(synset_url, synset_name)
    with open(synset_name) as f:
        synset = eval(f.read())

    # Read raw image and preprocess into the format ResNet can work on.
    img_name = "cat.png"
    download("https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true",
             img_name)
    image = Image.open(img_name).resize((224, 224))
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = tvm.nd.array(image.astype(dtype))

    block = get_model("resnet18_v1", pretrained=True)
    func, params = relay.frontend.from_mxnet(block,
                                             shape={"data": image.shape})

    with micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX):
        mod = relay_micro_build(func, TOOLCHAIN_PREFIX, params=params)
        # Execute with `image` as the input.
        mod.run(data=image)
        # Get outputs.
        tvm_output = mod.get_output(0)

        prediction_idx = np.argmax(tvm_output.asnumpy()[0])
        prediction = synset[prediction_idx]
        assert prediction == "tiger cat"


if __name__ == "__main__":
    # test_alloc()
    # test_add()
    # test_workspace_add()
    # test_graph_runtime()
    # test_multiple_modules()
    # test_interleave_sessions()
    test_resnet_random()
