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
import numpy as np

@tvm.register_extension
class MyTensorView(object):
    _tvm_tcode = tvm.TypeCode.DLTENSOR_HANDLE
    def __init__(self, arr):
        self.arr = arr

    @property
    def _tvm_handle(self):
        return self.arr._tvm_handle

def test_dltensor_compatible():
    dtype = 'int64'
    n = te.var('n')
    Ab = tvm.tir.decl_buffer((n,), dtype)
    i = te.var('i')
    ib = tvm.tir.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    with ib.for_range(0, n - 1, "i") as i:
        A[i + 1] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "arange"))
    f = tvm.build(mod, target="stackvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    aview = MyTensorView(a)
    f(aview)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))

if __name__ == "__main__":
    test_dltensor_compatible()
