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

def test_lower_rfactor():
    n = te.size_var("n")
    m = te.size_var("m")
    A = te.placeholder((n, m), name='A')
    k = te.reduce_axis((0, m), "k")
    B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B.op].bind(xo, te.thread_axis("blockIdx.x"))
    s[B.op].bind(xi, te.thread_axis("threadIdx.y"))
    s[B].bind(s[B].op.reduce_axis[0], te.thread_axis("threadIdx.x"))
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    fapi = tvm.lower(s, [A, B])

def test_dependent_output_shape():
    n, m, x = te.size_var('n'), te.size_var('m'), te.size_var('x')
    A = te.placeholder((n, m))
    B = te.compute((m, n//x), lambda i, j: A[i,j] , name='B')
    s = te.create_schedule(B.op)
    mod = tvm.build(s, [A, B, x])

def test_split_uneven_unique_likely():
    a = te.placeholder((16, 16),)
    b = te.placeholder((16, 16),)
    c = te.compute((16, 16), lambda x, y: a[x, y] + b[x, y])

    x, y = c.op.axis
    sch = te.create_schedule(c.op)
    xo, xi = sch[c].split(x, 5)
    stmt = tvm.lower(sch, [a, b, c], simple_mode=True)
    assert isinstance(stmt.body.body.body, tvm.tir.stmt.IfThenElse)
    assert str(stmt.body.body.body).count("likely") == 1

if __name__ == "__main__":
    test_lower_rfactor()
    test_dependent_output_shape()
    test_split_uneven_unique_likely()
