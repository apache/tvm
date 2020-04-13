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
import tvm
from tvm import te
import ctypes
import numpy as np


def MakeAPILegacy(stmt, name, args, num_unpacked_args, noalias):
    """Legacy adapter to create a API"""
    f = tvm.tir.PrimFunc(args, stmt).with_attr(
        "global_symbol", tvm.runtime.String(name))
    f = f.with_attr("tir.is_entry_func", True)
    if noalias:
        f = f.with_attr("tir.noalias", True)
    mod = tvm.IRModule.from_expr(f)
    return tvm.tir.transform.MakePackedAPI()(mod)


def test_static_callback():
    dtype = 'int64'
    n = te.size_var('n')
    Ab = tvm.tir.decl_buffer((n, ), dtype)
    i = te.size_var('i')
    ib = tvm.tir.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    cp = te.thread_axis((0, 1), "cop")
    finit = tvm.tir.StringImm("TVMBackendRunOnce")
    ib.scope_attr(cp, "coproc_uop_scope", finit)
    with ib.for_range(0, n, "i", for_type="parallel") as i:
        A[i] = A[i] + 1
    stmt = ib.get()
    fapi = tvm.testing.MakeAPILegacy(stmt, "ramp", [Ab], 0, True)
    f = tvm.driver.build(fapi, target="llvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)
    f(a)
    np.testing.assert_equal(a.asnumpy(), np.ones(a.shape[0]))

def test_static_init():
    dtype = 'int64'
    n = te.size_var('n')
    Ab = tvm.tir.decl_buffer((n, ), dtype)
    i = te.size_var('i')
    ib = tvm.tir.ir_builder.create()
    handle = tvm.tir.call_intrin("handle", "tvm_static_handle")
    ib.emit(
        tvm.tir.call_packed("test_static_callback", handle, Ab))

    @tvm.register_func("test_static_callback")
    def test_cb(sh, A):
        assert isinstance(sh, ctypes.c_void_p)
        return sh

    stmt = ib.get()
    fapi = tvm.testing.MakeAPILegacy(stmt, "ramp", [Ab], 0, True)
    f = tvm.driver.build(fapi, target="llvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)


if __name__ == "__main__":
    test_static_callback()
    test_static_init()
