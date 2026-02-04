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
from tvm.script import ir as I, tir as T
import ctypes
import numpy as np


def test_static_callback():
    @I.ir_module
    class Module:
        @T.prim_func
        def ramp(A: T.handle):
            T.func_attr({"global_symbol": "ramp"})
            n = T.int64()
            Ab = T.match_buffer(A, (n,), "int64")
            # coproc_uop_scope with TVMBackendRunOnce ensures body runs only once
            with T.attr(
                T.iter_var(T.int32(), (0, 1), "DataPar", "cop"),
                "coproc_uop_scope",
                "TVMBackendRunOnce",
            ):
                for i in T.parallel(n):
                    Ab[i] = Ab[i] + T.int64(1)

    mod = Module
    f = tvm.driver.build(mod, target="llvm")
    a = tvm.runtime.tensor(np.zeros(10, dtype="int64"))
    f(a)
    f(a)
    np.testing.assert_equal(a.numpy(), np.ones(a.shape[0]))


def test_static_init():
    @tvm.register_global_func("test_static_callback")
    def test_cb(sh, A):
        assert isinstance(sh, ctypes.c_void_p)
        return sh

    @I.ir_module
    class Module:
        @T.prim_func
        def ramp(A: T.handle):
            T.func_attr({"global_symbol": "ramp"})
            n = T.int64()
            Ab = T.match_buffer(A, (n,), "int64")
            T.call_packed(
                "test_static_callback",
                T.call_intrin("handle", "tir.tvm_static_handle"),
                Ab.data,
            )

    mod = Module
    f = tvm.driver.build(mod, target="llvm")
    a = tvm.runtime.tensor(np.zeros(10, dtype="int64"))
    f(a)


if __name__ == "__main__":
    test_static_callback()
    test_static_init()
