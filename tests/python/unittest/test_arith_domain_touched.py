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
from tvm.script import tir as T


@T.prim_func
def scalar_func(a: T.handle, b: T.handle):
    m = T.var("int32")
    n = 100
    A = T.match_buffer(a, (n, m))
    B = T.match_buffer(b, (n, m))

    for i, j in T.grid(n, m):
        A[i, j] = B[i - 1, j + 1] + A[i - 1, j - 1]


def test_domain_touched():
    func = scalar_func
    a, b = [func.buffer_map[var] for var in func.params]
    ir = func.body

    a_domain_r = tvm.arith._ffi_api.DomainTouched(ir, a, True, False)

    assert a_domain_r[0].min.value == -1
    assert a_domain_r[0].extent.value == 100
    assert a_domain_r[1].min.value == -1
    assert a_domain_r[1].extent.name == "m"

    a_domain_w = tvm.arith._ffi_api.DomainTouched(ir, a, False, True)
    assert a_domain_w[0].min.value == 0
    assert a_domain_w[0].extent.value == 100
    assert a_domain_w[1].min.value == 0
    assert a_domain_w[1].extent.name == "m"

    a_domain_rw = tvm.arith._ffi_api.DomainTouched(ir, a, True, True)
    assert a_domain_rw[0].min.value == -1
    assert a_domain_rw[0].extent.value == 101
    assert a_domain_rw[1].min.value == -1
    assert isinstance(a_domain_rw[1].extent, tvm.tir.Add)
    assert a_domain_rw[1].extent.a.name == "m"
    assert a_domain_rw[1].extent.b.value == 1

    b_domain_r = tvm.arith._ffi_api.DomainTouched(ir, b, True, False)
    assert b_domain_r
    assert b_domain_r[0].min.value == -1
    assert b_domain_r[0].extent.value == 100
    assert b_domain_r[1].min.value == 1
    assert b_domain_r[1].extent.name == "m"

    b_domain_w = tvm.arith._ffi_api.DomainTouched(ir, b, False, True)
    assert isinstance(b_domain_w, tvm.container.Array)
    assert len(b_domain_w) == 0


def test_domain_touched_vector():
    m = tvm.runtime.convert(128)

    @T.prim_func
    def func(a: T.handle, b: T.handle):
        n = T.var("int32")
        A = T.match_buffer(a, (n * m,))
        B = T.match_buffer(b, (n * m,))

        for i in T.serial(n):
            A[i * m : (i + 1) * m : 1] = A[i * m : (i + 1) * m : 1] + B[i * m : (i + 1) * m : 1]

    a, b = [func.buffer_map[var] for var in func.params]

    assert tvm.arith._ffi_api.DomainTouched(func.body, a, True, False)[0].extent.value == 128
    assert tvm.arith._ffi_api.DomainTouched(func.body, a, True, False)[0].extent.value == 128
    assert tvm.arith._ffi_api.DomainTouched(func.body, a, True, True)[0].extent.value == 128
    assert tvm.arith._ffi_api.DomainTouched(func.body, b, True, False)[0].extent.value == 128
    assert tvm.arith._ffi_api.DomainTouched(func.body, b, True, False)[0].extent.value == 128


if __name__ == "__main__":
    test_domain_touched()
