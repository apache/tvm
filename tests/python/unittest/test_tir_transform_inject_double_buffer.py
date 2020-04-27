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

def test_double_buffer():
    dtype = 'int64'
    n = 100
    m = 4
    tx = te.thread_axis("threadIdx.x")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    ib.scope_attr(tx, "thread_extent", 1)
    with ib.for_range(0, n) as i:
        B = ib.allocate("float32", m, name="B", scope="shared")
        with ib.new_scope():
            ib.scope_attr(B.asobject(), "double_buffer_scope", 1)
            with ib.for_range(0, m) as j:
                B[j] = A[i * 4 + j]
        with ib.for_range(0, m) as j:
            C[j] = B[j] + 1

    stmt = ib.get()
    mod = tvm.IRModule({
        "db" : tvm.tir.PrimFunc([A.asobject(), C.asobject()], stmt)
    })

    opt = tvm.transform.Sequential(
        [tvm.tir.transform.InjectDoubleBuffer(2),
         tvm.tir.transform.Simplify()])
    mod = opt(mod)
    stmt = mod["db"].body

    assert isinstance(stmt.body.body, tvm.tir.Allocate)
    assert stmt.body.body.extents[0].value == 2

    f = tvm.tir.transform.ThreadSync("shared")(mod)["db"]
    count = [0]
    def count_sync(op):
        if isinstance(op, tvm.tir.Call) and op.name == "tvm_storage_sync":
            count[0] += 1
    tvm.tir.stmt_functor.post_order_visit(f.body, count_sync)
    assert count[0] == 4


if __name__ == "__main__":
    test_double_buffer()
