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

def test_coproc_lift():
    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    cp = te.thread_axis((0, 1), "cop")
    value = tvm.tir.StringImm("xxx")

    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            ib.scope_attr(cp, "coproc_uop_scope", value)
            A[i] = A[i] + 1
        with ib.if_scope(i.equal(0)):
            with ib.for_range(0, 10, name="j") as j:
                ib.scope_attr(cp, "coproc_uop_scope", value)
                A[j] = A[j] + 2
                A[j] = A[j] + 3
                A[j] = A[j] + 3
    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.LiftAttrScope("coproc_uop_scope")(mod)["main"].body

    assert body.body.body.node == cp

    # only able to lift to the common pattern of the last two fors.
    ib = tvm.tir.ir_builder.create()
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A[j] = A[j] + 1
        with ib.for_range(0, 10, name="j") as j:
            ib.scope_attr(cp, "coproc_uop_scope", value)
            A[i] = A[i] + 1
        with ib.for_range(0, 10, name="j") as j:
            ib.scope_attr(cp, "coproc_uop_scope", value)
            A[i] = A[i] + 2

    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.LiftAttrScope("coproc_uop_scope")(mod)["main"].body

    assert body.body.body.body[1].node == cp
    assert len(body.body.body.body) == 2

if __name__ == "__main__":
    test_coproc_lift()
