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

def test_domain_touched():
    i = te.var('i')
    j = te.var('j')
    n = tvm.runtime.convert(100)
    m = te.var('m')

    a = tvm.tir.decl_buffer((n, m), name='a')
    b = tvm.tir.decl_buffer((n, m), name='b')


    ir = tvm.tir.For(
            i, 0, n, 0, 0,
            tvm.tir.For(j, 0, m, 0, 0,
                tvm.tir.BufferStore(
                    a,
                    tvm.tir.BufferLoad(b, [i - 1, j + 1]) +
                    tvm.tir.BufferLoad(a, [i - 1, j - 1]),
                    [i, j]
                )
            )
    )

    a_domain_r = tvm.arith._ffi_api.DomainTouched(ir, a, True, False)

    assert a_domain_r[0].min.value == -1
    assert a_domain_r[0].extent.value == 100
    assert a_domain_r[1].min.value == -1
    assert a_domain_r[1].extent.name == 'm'

    a_domain_w = tvm.arith._ffi_api.DomainTouched(ir, a, False, True)
    assert a_domain_w[0].min.value == 0
    assert a_domain_w[0].extent.value == 100
    assert a_domain_w[1].min.value == 0
    assert a_domain_w[1].extent.name == 'm'

    a_domain_rw= tvm.arith._ffi_api.DomainTouched(ir, a, True, True)
    assert a_domain_rw[0].min.value == -1
    assert a_domain_rw[0].extent.value == 101
    assert a_domain_rw[1].min.value == -1
    assert isinstance(a_domain_rw[1].extent, tvm.tir.Add)
    assert a_domain_rw[1].extent.a.name == 'm'
    assert a_domain_rw[1].extent.b.value == 1

    b_domain_r = tvm.arith._ffi_api.DomainTouched(ir, b, True, False)
    assert b_domain_r
    assert b_domain_r[0].min.value == -1
    assert b_domain_r[0].extent.value == 100
    assert b_domain_r[1].min.value == 1
    assert b_domain_r[1].extent.name == 'm'

    b_domain_w = tvm.arith._ffi_api.DomainTouched(ir, b, False, True)
    assert isinstance(b_domain_w, tvm.container.Array)
    assert len(b_domain_w) == 0

if __name__ == "__main__":
    test_domain_touched()
