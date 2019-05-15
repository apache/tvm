import tvm
import os
import logging
import subprocess
import time

import numpy as np
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro

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


def test_farts():
    nn = 10
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name="A")
    # B = tvm.placeholder((n,), name="B")
    # B = tvm.compute(B.shape, lambda *i: A(*i) + 1, name="B")
    # C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    # s = tvm.create_schedule(C.op)

    init_lib_path = micro.get_init_lib()
    micro.init("host", init_lib_path)
    m = tvm.module.load("farts.obj", "micro_dev")
    ctx = tvm.micro_dev(0)
    fadd_workspace = m["fused_add"]
    n = nn
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=A.dtype), ctx)
    print(a)
    print(c)
    fadd_workspace(a, c)
    print(a)
    print(c)
    print()

    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + 1.0)


def test_graph_runtime():
    dtype = "float32"
    shape = (10,)

    # build relay program
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    y = relay.const(1.0)
    z = relay.add(y, y)
    ayy = relay.add(x, z)
    func = relay.Function([x], ayy)
    graph, lib, params = relay.build(func, target="c", params={})
    print(graph)

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
        line = mod_src[i]
        if line.endswith("{") and any([s in line for s in ["dev_type", "device_type", "device_id"]]):
            while not mod_src[i].strip().endswith("}"):
                mod_src.pop(i)
            mod_src.pop(i)
        elif "return -1;" in line:
            mod_src[i] = f"return -{curr_return_err};"
            curr_return_err += 1
            i += 1
        else:
            i += 1
    mod_src = "\n".join(mod_src)
    print(mod_src)

    # with open("farts.c", "r") as f:
    #     mod_src = f.read()
    # print(mod_src)
    # save it to temp file
    src_dso = temp.relpath("dev_lib.c")
    with open(src_dso, "w") as f:
        f.write(mod_src)

    # compile to object file
    lib_dso = temp.relpath("dev_lib.obj")
    tvm_home = os.getenv("TVM_HOME")
    # retcode = subprocess.call(["gcc", "-c", "-g", "-Og", "-o", lib_dso, src_dso, f"-I{tvm_home}/include", f"-I{tvm_home}/3rdparty/dlpack/include"])
    retcode = subprocess.call(["gcc", "-c", "-g", "-O0", "-o", lib_dso, src_dso, f"-I{tvm_home}/include", f"-I{tvm_home}/3rdparty/dlpack/include"])
    assert retcode == 0

    micro.init("host", micro.get_init_lib())
    micro_lib = tvm.module.load(lib_dso, "micro_dev")
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_lib, ctx)

    # # compile to object file
    # lib_dso = temp.relpath("dev_lib.o")
    # tvm_home = os.getenv("TVM_HOME")
    # subprocess.call(["gcc", "-fPIC", "-c", "-g", "-Og", "-o", lib_dso, src_dso, f"-I{tvm_home}/include", f"-I{tvm_home}/3rdparty/dlpack/include"])

    # host_lib = tvm.module.load(lib_dso)
    # ctx = tvm.cpu(0)
    # mod = graph_runtime.create(graph, host_lib, ctx)

    print(f"params: {params}")
    x_in = np.random.uniform(size=shape[0]).astype(dtype)
    print(f"x_in: {x_in}")
    print(f"mod: {mod}")
    mod.set_input(**params)
    # mod.set_input("x", x_in)
    print("running module...")
    mod.run(x=x_in)
    print("finished running")
    out = mod.get_output(0, tvm.nd.empty(shape)).asnumpy()
    print(f"output: {out}")



if __name__ == "__main__":
    # test_add()
    # test_workspace_add()
    # test_farts()
    test_graph_runtime()
