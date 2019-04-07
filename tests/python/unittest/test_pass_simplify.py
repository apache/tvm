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
import numpy
from tvm import comm_reducer
from tvm.ir_pass import Simplify, CanonicalSimplify, Equal

def test_simplify():
    """Not yet working, mock design"""
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    j = tvm.var('j')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 2, n, 0, 0,
        tvm.make.For(j, 0, n, 0, 0,
                     tvm.make.IfThenElse(
                         tvm.make.LT(i + 2, n),
                         tvm.make.Store(Ab.data,
                                        tvm.make.Load(dtype, Ab.data, i + 4) + 1,
                                        (j + 1) * 4 - 4 * j + i),
                         None)))
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)


def test_basic():
    m = tvm.var('m')
    ret = tvm.ir_pass.CanonicalSimplify(tvm.make.Evaluate(m-1))
    assert str(ret.value) == "(m - 1)"


def test_bound():
    m = tvm.var('m')
    vrange = tvm.convert({m: tvm.Range(tvm.const(0, "int32"), tvm.const(10, "int32"))})
    ret = tvm.ir_pass.Simplify(m % 10, vrange)
    assert ret == m


if __name__ == "__main__":
    test_bound()
    test_basic()
    test_simplify()
