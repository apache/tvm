import logging
import os

import numpy as np

import tvm
from tvm import rpc
from tvm import micro
from tvm.contrib import graph_runtime, util

assert tvm.module.enabled("rpc")
assert tvm.module.enabled("micro_dev")

def create_and_load_micro_mod(remote, c_mod, name, toolchain_prefix):
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
    lib_obj_path = temp_dir.relpath(f'{name}.obj')
    c_mod.export_library(
            lib_obj_path,
            fcompile=tvm.micro.cross_compiler(toolchain_prefix, micro.LibType.OPERATOR))
    remote.upload(lib_obj_path)
    micro_mod = remote.load_module(os.path.basename(lib_obj_path))
    #micro_mod = tvm.module.load(lib_obj_path)
    return micro_mod


def reset_gdbinit():
    with open('/home/pratyush/Code/nucleo-interaction-from-scratch/.gdbinit', 'w') as f:
        gdbinit_contents = (
"""layout asm
target remote localhost:3333

#print "(*((TVMValue*) utvm_task.arg_values)).v_handle"
#print (*((TVMValue*) utvm_task.arg_values)).v_handle
#print "*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))"
#print *((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))
#print "((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[0]"
#print ((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[0]
#print "((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[1]"
#print ((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[1]
#break UTVMMain
#break UTVMDone
#jump UTVMMain""")
        f.write(gdbinit_contents)


#DEVICE_TYPE = 'host'
#TOOLCHAIN_PREFIX = ''
DEVICE_TYPE = 'openocd'
TOOLCHAIN_PREFIX = 'arm-none-eabi-'

def test_utvm_session():
    server = rpc.Server("localhost", key="micro")
    remote = rpc.connect(server.host, server.port, key="micro")
    utvm_sess_create = remote.get_function("micro.create_session")
    sess = utvm_sess_create(DEVICE_TYPE, TOOLCHAIN_PREFIX)
    sess['enter']()
    sess['exit']()


def test_rpc_alloc():
    shape = (1024,)
    dtype = "float32"

    server = rpc.Server("localhost", key="micro")
    remote = rpc.connect(server.host, server.port, key="micro")
    utvm_sess_create = remote.get_function("micro.create_session")
    sess = utvm_sess_create(DEVICE_TYPE, TOOLCHAIN_PREFIX)
    sess['enter']()

    ctx = remote.micro_dev(0)
    np_tensor = np.random.uniform(size=shape).astype(dtype)
    micro_tensor = tvm.nd.array(np_tensor, ctx)
    tvm.testing.assert_allclose(np_tensor, micro_tensor.asnumpy())

    sess['exit']()


def test_rpc_add():
    shape = (1024,)
    dtype = "float32"

    reset_gdbinit()

    server = rpc.Server("localhost", key="micro")
    remote = rpc.connect(server.host, server.port, key="micro")
    utvm_sess_create = remote.get_function("micro.create_session")
    sess = utvm_sess_create(DEVICE_TYPE, TOOLCHAIN_PREFIX)

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    func_name = 'fadd'
    c_mod = tvm.build(s, [A, B, C], target='c', name=func_name)

    sess['enter']()

    micro_mod = create_and_load_micro_mod(remote, c_mod, 'fadd', TOOLCHAIN_PREFIX)
    micro_func = micro_mod[func_name]

    ctx = remote.micro_dev(0)
    a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
    print(a)
    b = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
    print(b)
    c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
    print(c)
    micro_func(a, b, c)
    print(c)

    sess['exit']()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #test_utvm_session()
    #test_rpc_alloc()
    test_rpc_add()
