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
from tvm import tir
from tvm.ir.transform import PassContext

def test_scalar_add():
    a = tir.Var("a", "float32")
    b = tir.Var("b", "float32")
    c = a + b
    c = tir.call_intrin("float32", "tir.ret", c)
    c = tir.Evaluate(c)
    func = tir.PrimFunc([a, b], c)
    func = func.with_attr("global_symbol", "main")
    pass_ctx = PassContext.current()
    if pass_ctx.config.get("tir.noalias", True):
        func = func.with_attr("tir.noalias", True)
    mod = tvm.IRModule({'main': func})
    func = tvm.build(mod)
    out = func(1.0, 2.0)
    assert out == 3.0

if __name__ == "__main__":
    test_scalar_add()
