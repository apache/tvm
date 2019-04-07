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
import ctypes
import numpy as np

def test_static_callback():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    ib = tvm.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    cp = tvm.thread_axis((0, 1), "cop")
    finit = tvm.make.StringImm("TVMBackendRunOnce")
    ib.scope_attr(cp, "coproc_uop_scope", finit)
    with ib.for_range(0, n, "i", for_type="parallel") as i:
        A[i] = A[i] + 1
    stmt = ib.get()
    fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 0, True)
    fapi = tvm.ir_pass.LowerTVMBuiltin(fapi)
    f = tvm.codegen.build_module(fapi, "llvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)
    f(a)
    np.testing.assert_equal(a.asnumpy(), np.ones(a.shape[0]))

def test_static_init():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    ib = tvm.ir_builder.create()
    handle = tvm.call_intrin("handle", "tvm_static_handle")
    ib.emit(
        tvm.call_packed("test_static_callback", handle, Ab))

    @tvm.register_func("test_static_callback")
    def test_cb(sh, A):
        assert isinstance(sh, ctypes.c_void_p)
        return sh

    stmt = ib.get()
    fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 0, True)
    fapi = tvm.ir_pass.LowerTVMBuiltin(fapi)
    f = tvm.codegen.build_module(fapi, "llvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)


if __name__ == "__main__":
    test_static_callback()
    test_static_init()
