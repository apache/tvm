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

def nop():
    return tvm.tir.Evaluate(0)

def test_remove_no_op():
    i = tvm.var('i')
    j = tvm.var('j')
    k = tvm.var('k')
    m = tvm.var('m')
    n = tvm.var('n')
    dtype = 'int64'
    Ab = tvm.decl_buffer((n, ), dtype)
    stmt = tvm.tir.For(
        i, 0, 4, 0, 0,
        tvm.tir.For(
            j, 0, n, 0, 0,
            tvm.tir.For(
                k, 0, m, 0, 0,
                tvm.tir.IfThenElse(
                    (i*m+j+k < n), tvm.tir.Evaluate(m), tvm.tir.Evaluate(n)))))
    ret = tvm.ir_pass.RemoveNoOp(stmt)
    assert(isinstance(ret, tvm.tir.Evaluate))
    store = tvm.tir.Store(Ab.data,
                           tvm.tir.Load(dtype, Ab.data, i) + 1,
                           i + 1)
    stmt2 = tvm.tir.SeqStmt([nop(), tvm.tir.SeqStmt([store, nop()])])
    assert(tvm.ir_pass.RemoveNoOp(stmt2) == store)
    # remove zero extent loop
    stmt3 = tvm.tir.For(i, 0, 0, 0, 0, store)
    ret = tvm.ir_pass.RemoveNoOp(stmt3)
    assert(isinstance(ret, tvm.tir.Evaluate))


if __name__ == "__main__":
    test_remove_no_op()
