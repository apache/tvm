import contextlib
import copy
import glob
import os
import pty
import sys
import subprocess
import textwrap

import numpy as np

import tvm
import tvm.relay
import tvm.micro

from tvm.topi.util import get_const_tuple
from tvm.topi.testing import conv2d_nchw_python

BUILD = True
DEBUG = False

# TODO(weberlo) fix bug with sessions not being reusable
# ADD_SESS = None
# IDENT_SESS = None

TARGET = tvm.target.target.micro('host')

def _make_sess_from_op(op_name, sched, arg_bufs):
  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(sched, arg_bufs, TARGET, target_host=TARGET, name=op_name)

  return _make_session(mod)


def _make_session(mod):
  workspace = tvm.micro.Workspace(debug=True)

  compiler = tvm.micro.DefaultCompiler(target=TARGET)
  opts = tvm.micro.DefaultOptions(os.path.join(tvm.micro.CRT_ROOT_DIR, 'host'))

  micro_binary = tvm.micro.build_static_runtime(
    # the x86 compiler *expects* you to give the exact same dictionary for both
    # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
    # the binary compiler is expecting those mutations to be in bin_opts.
    # TODO(weberlo) fix this very bizarre behavior
    workspace, compiler, mod, lib_opts=opts['bin_opts'], bin_opts=opts['bin_opts'])

  flasher_kw = {
    'debug': DEBUG,
  }
  flasher = compiler.Flasher(**flasher_kw)
  return tvm.micro.Session(binary=micro_binary, flasher=flasher)


def _make_add_sess():
  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')
  sched = tvm.te.create_schedule(C.op)
  return _make_sess_from_op('add', sched, [A, B, C])


def _make_ident_sess():
  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.compute(A.shape, lambda i: A[i], name='B')
  sched = tvm.te.create_schedule(B.op)
  return _make_sess_from_op('ident', sched, [A, B])


def test_compile_runtime():
  """Test compiling the on-device runtime."""
  with _make_add_sess() as sess:
    A_data = tvm.nd.array(np.array([2, 3], dtype='int8'), ctx=sess.context)
    assert (A_data.asnumpy() == np.array([2, 3])).all()
    B_data = tvm.nd.array(np.array([4], dtype='int8'), ctx=sess.context)
    assert (B_data.asnumpy() == np.array([4])).all()
    C_data = tvm.nd.array(np.array([0, 0], dtype='int8'), ctx=sess.context)
    assert (C_data.asnumpy() == np.array([0, 0])).all()

    print('get system lib')
    system_lib = sess.get_system_lib()
    print('got system lib', system_lib)
    system_lib.get_function('add')(A_data, B_data, C_data)
    print('got data!', C_data.asnumpy())
    assert (C_data.asnumpy() == np.array([6, 7])).all()


if __name__ == '__main__':
  test_compile_runtime()
