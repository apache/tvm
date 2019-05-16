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

# adds two arrays and stores result into third array
def test_add():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name="A")
    B = tvm.placeholder((n,), name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    init_lib_path = micro.get_init_lib()
    micro.init("host", init_lib_path)
    m = tvm.module.load("fadd.obj", "micro_dev")
    ctx = tvm.micro_dev(0)
    fadd = m["fadd"]
    n = nn
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    print(a)
    print(b)
    print(c)
    fadd(a, b, c)
    print(a)
    print(b)
    print(c)
    print()

    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_workspace_add():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name="A")
    B = tvm.placeholder((n,), name="B")
    B = tvm.compute(B.shape, lambda *i: A(*i) + 1, name="B")
    C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    s = tvm.create_schedule(C.op)

    init_lib_path = micro.get_init_lib()
    micro.init("host", init_lib_path)
    m = tvm.module.load("fadd_workspace.obj", "micro_dev")
    ctx = tvm.micro_dev(0)
    fadd_workspace = m["fadd_workspace"]
    n = nn
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    print(a)
    print(c)
    fadd_workspace(a, c)
    print(a)
    print(c)
    print()

    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + 2.0)


def micro_module(func: relay.Function, params={}):
    with tvm.build_config(disable_vectorize=True):
        graph, lib, params = relay.build(func, target="c", params=params)

    temp = util.tempdir()

    # modify source from C codegen to include device library header
    mod_src = lib.get_source().split("\n")
    # TODO(weberlo): either make a new "micro_dev" codegen target that
    # properly wraps the C codegen or search for the end of the includes.
    mod_src.insert(2, "#include \"tvm/runtime/utvm_device_lib.h\"")
    # TODO(weberlo): this shit is a mega hack
    i = 0
    curr_return_err = 1
    while i < len(mod_src):
        if mod_src[i].endswith("{") and any([s in mod_src[i] for s in ["dev_type", "device_type"]]):
            while not mod_src[i].strip().endswith("}"):
                mod_src.pop(i)
            mod_src.pop(i)
        elif "return -1;" in mod_src[i]:
            mod_src[i] = mod_src[i].replace("-1", f"-{curr_return_err}")
            curr_return_err += 1
            i += 1
        else:
            i += 1
    mod_src = "\n".join(mod_src)

    # save it to temp file
    src_dso = temp.relpath("dev_lib.c")
    with open(src_dso, "w") as f:
        f.write(mod_src)

    # compile to object file
    lib_dso = temp.relpath("dev_lib.obj")
    tvm_home = os.getenv("TVM_HOME")
    cmd = ["g++", "-fno-stack-protector", "-c", "-g", "-O0", "-o", lib_dso, src_dso, f"-I{tvm_home}/include", f"-I{tvm_home}/3rdparty/dlpack/include"]
    print(f"compiling with \"{cmd}\"")
    retcode = subprocess.call(cmd)
    assert retcode == 0

    micro.init("host", micro.get_init_lib())
    micro_lib = tvm.module.load(lib_dso, "micro_dev")
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_lib, ctx)
    return mod, params


def test_graph_runtime():
    dtype = "float32"
    shape = (10,)

    # build relay program
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    y = relay.const(1.0)
    xx = relay.multiply(x, x)
    z = relay.add(xx, y)
    func = relay.Function([x], z)

    mod, params = micro_module(func)

    x_in = np.random.uniform(size=shape[0]).astype(dtype)
    mod.set_input(**params)
    mod.run(x=x_in)
    out = mod.get_output(0, tvm.nd.empty(shape)).asnumpy()
    print(f"output: {out}")


def test_resnet():
    resnet_func, orig_params = resnet.get_workload(num_classes=10, num_layers=18, image_shape=(3, 32, 32))
    # TODO(weberlo): use `resnet_func` once we have libc support.
    # remove the final softmax layer, because uTVM does not currently support it
    resnet_func_no_sm = relay.Function(resnet_func.params, resnet_func.body.args[0], resnet_func.ret_type)
    mod, params = micro_module(resnet_func_no_sm, params=orig_params)
    mod.set_input(**params)
    # generate random input
    data = np.random.uniform(size=mod.get_input(0).shape)
    mod.run(data=data)
    print(f"output: {mod.get_output(0)}")


if __name__ == "__main__":
    # test_add()
    # test_workspace_add()
    # test_graph_runtime()
    test_resnet()
