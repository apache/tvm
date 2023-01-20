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
import pytest
import tvm
from tvm import tir, ir


def test_convert_ssa():
    dtype = "int32"
    zero = tir.const(0)
    nop = tir.Evaluate(zero)
    var_type = ir.PointerType(ir.PrimType(dtype))
    v = tir.Var("i1", var_type)
    buf = tir.decl_buffer([16], dtype=dtype, data=v)
    let = tir.LetStmt(v, v, nop)
    load = tir.Evaluate(tir.BufferLoad(buf, [zero]))
    seq = tir.SeqStmt([let, let, load])
    func = tir.PrimFunc([], seq)
    mod = tvm.IRModule({"main": func})
    mod = tir.transform.InjectVirtualThread()(
        mod
    )  # Use pass InjectVirtualThread to invoke ConvertSSA


if __name__ == "__main__":
    tvm.testing.main()
