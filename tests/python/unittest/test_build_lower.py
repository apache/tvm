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

def test_lower_rfactor():
    n = tvm.var("n")
    m = tvm.var("m")
    A = tvm.placeholder((n, m), name='A')
    k = tvm.reduce_axis((0, m), "k")
    B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name="B")
    s = tvm.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B.op].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[B.op].bind(xi, tvm.thread_axis("threadIdx.y"))
    s[B].bind(s[B].op.reduce_axis[0], tvm.thread_axis("threadIdx.x"))
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    fapi = tvm.lower(s, [A, B])

def test_dependent_output_shape():
    n, m, x = tvm.var('n'), tvm.var('m'), tvm.var('x')
    A = tvm.placeholder((n, m))
    B = tvm.compute((m, n//x), lambda i, j: A[i,j] , name='B')
    s = tvm.create_schedule(B.op)
    mod = tvm.build(s, [A, B, x])

if __name__ == "__main__":
    test_lower_rfactor()
    test_dependent_output_shape()
