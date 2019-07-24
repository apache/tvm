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

def test_stmt_simplify():
    ib = tvm.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.if_scope(i < 12):
            A[i] = C[i]

    body = tvm.stmt.LetStmt(n, 10, ib.get())
    body = tvm.ir_pass.CanonicalSimplify(body)
    assert isinstance(body.body, tvm.stmt.Store)


if __name__ == "__main__":
    test_stmt_simplify()
