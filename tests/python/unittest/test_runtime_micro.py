import tvm
import os
import logging
import subprocess
import time

import numpy as np
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.relay.testing import resnet

# TODO(weberlo): document somewhere that utvm object files need to have an
# `.obj` instead of an `.o` extension, because the `.o` suffix triggers a code
# path we don't want in `module.load`.

# TODO(weberlo): We should just move this entire function into `tvm.micro`.
def compile_lib(lib_mod, temp_dir):
    # save source to temp file
    lib_src_path = temp_dir.relpath("dev_lib.c")
    mod_src = lib_mod.get_source()
    with open(lib_src_path, "w") as f:
        f.write(mod_src)
    # compile to object file
    # TODO(weberlo): it'd be ideal if we didn't need to pass a compile command
    # here, but rather the device type, or just the library module.
    lib_obj_path = micro.create_micro_lib("gcc", lib_src_path)
    return lib_obj_path


def relay_micro_build(func: relay.Function, params={}):
    """Create a graph runtime module with a micro device context."""
    with tvm.build_config(disable_vectorize=True):
        with relay.build_config(opt_level=3):
            graph, lib_mod, params = relay.build(func, target="c", params=params)

    temp_dir = util.tempdir()
    lib_obj_path = compile_lib(lib_mod, temp_dir)

    micro.init("host")
    micro_lib = tvm.module.load(lib_obj_path, "micro_dev")
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_lib, ctx)
    return mod, params


def test_add():
    """Test a program which performs addition."""
    shape = (1024,)
    dtype = "float32"

    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd"
    lib_mod = tvm.build(s, [A, B, C], target="c", name=func_name)
    temp_dir = util.tempdir()
    lib_obj_path = compile_lib(lib_mod, temp_dir)

    micro.init("host")
    micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
    ctx = tvm.micro_dev(0)
    micro_func = micro_mod[func_name]
    a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
    c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
    micro_func(a, b, c)

    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_workspace_add():
    """Test a program which uses a workspace."""
    # adds two arrays and stores result into third array

    shape = (1024,)

    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A")
    B = tvm.placeholder(tvm_shape, name="B")
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd_two_workspace"
    lib_mod = tvm.build(s, [A, C], target="c", name=func_name)
    temp_dir = util.tempdir()
    lib_obj_path = compile_lib(lib_mod, temp_dir)

    micro.init("host")
    micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
    ctx = tvm.micro_dev(0)
    micro_func = micro_mod[func_name]
    a = tvm.nd.array(np.random.uniform(size=shape).astype(A.dtype), ctx)
    c = tvm.nd.array(np.zeros(shape, dtype=C.dtype), ctx)
    micro_func(a, c)

    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + 2.0)


def test_graph_runtime():
    """Test a program which uses the graph runtime."""
    shape = (10,)
    dtype = "float32"

    # construct relay program
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(1.0))
    func = relay.Function([x], z)

    mod, params = relay_micro_build(func)

    mod.set_input(**params)
    x_in = np.random.uniform(size=shape[0]).astype(dtype)
    mod.run(x=x_in)
    result = mod.get_output(0).asnumpy()

    tvm.testing.assert_allclose(
        result, x_in * x_in + 1.0)


def test_resnet_random():
    """Test ResNet18 inference with random weights and inputs."""
    resnet_func, params = resnet.get_workload(num_classes=10, num_layers=18, image_shape=(3, 32, 32))
    # remove the final softmax layer, because uTVM does not currently support it
    resnet_func_no_sm = relay.Function(resnet_func.params, resnet_func.body.args[0], resnet_func.ret_type)
    # TODO(weberlo): use `resnet_func` once we have libc support.
    mod, params = relay_micro_build(resnet_func_no_sm, params=params)
    mod.set_input(**params)
    # generate random input
    data = np.random.uniform(size=mod.get_input(0).shape)
    mod.run(data=data)
    result = mod.get_output(0).asnumpy()
    # we gave a random input, so all we want is a result with some nonzero entries
    assert result.sum() != 0.0


def test_resnet_pretrained():
    """Test classification with a pretrained ResNet18 model."""
    # TODO(weberlo) there's a significant amount of overlap between here and
    # `tutorials/frontend/from_mxnet.py`.  Refactor pls.

    # some standard imports
    import mxnet as mx
    import numpy as np

    from mxnet.gluon.model_zoo.vision import get_model
    from mxnet.gluon.utils import download
    from PIL import Image
    from matplotlib import pyplot as plt

    dtype = "float32"

    block = get_model("resnet18_v1", pretrained=True)
    img_name = "cat.png"
    synset_url = "".join(["https://gist.githubusercontent.com/zhreshold/",
                          "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                          "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                          "imagenet1000_clsid_to_human.txt"])
    synset_name = "synset.txt"
    download("https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true", img_name)
    download(synset_url, synset_name)
    with open(synset_name) as f:
        synset = eval(f.read())
    image = Image.open(img_name).resize((224, 224))
    plt.imshow(image)
    plt.show()

    def transform_image(image):
        image = np.array(image) - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)

    shape_dict = {"data": x.shape}
    func, params = relay.frontend.from_mxnet(block, shape_dict)

    mod, params = relay_micro_build(func, params=params)

    # set inputs
    mod.set_input("data", tvm.nd.array(x.astype(dtype)))
    mod.set_input(**params)
    # execute
    mod.run()
    # get outputs
    tvm_output = mod.get_output(0)
    prediction_idx = np.argmax(tvm_output.asnumpy()[0])
    prediction = synset[prediction_idx]
    assert prediction == "tiger cat"


if __name__ == "__main__":
    test_add()
    test_workspace_add()
    test_graph_runtime()
    test_resnet_random()
    test_resnet_pretrained()
