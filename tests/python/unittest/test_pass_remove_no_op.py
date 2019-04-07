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

def test_remove_no_op():
    i = tvm.var('i')
    j = tvm.var('j')
    k = tvm.var('k')
    m = tvm.var('m')
    n = tvm.var('n')
    dtype = 'int64'
    Ab = tvm.decl_buffer((n, ), dtype)
    stmt = tvm.make.For(
        i, 0, 4, 0, 0,
        tvm.make.For(
            j, 0, n, 0, 0,
            tvm.make.For(
                k, 0, m, 0, 0,
                tvm.make.IfThenElse(
                    (i*m+j+k < n), tvm.make.Evaluate(m), tvm.make.Evaluate(n)))))
    ret = tvm.ir_pass.RemoveNoOp(stmt)
    assert(isinstance(ret, tvm.stmt.Evaluate))
    store = tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1,
                           i + 1)
    stmt2 = tvm.make.Block(stmt, store)
    assert(tvm.ir_pass.RemoveNoOp(stmt2) == store)


if __name__ == "__main__":
    test_remove_no_op()
