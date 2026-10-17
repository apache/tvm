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

"""Executable examples from the parser namespace introduction."""

import json
from types import SimpleNamespace

from tvm_tirx import ir
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.script.parser import make_decorator, register_namespace
from tvm_tirx.tirx.script import ir_builder as B


@T.function
def example(a: T.int32):
    T.evaluate(a + 1)


with IRBuilder() as builder:
    with B.function_():
        B.func_name_("example")
        a = B.arg_("a", B.int32)
        B.emit_(B.evaluate(a + 1))
built = builder.get()
ir.assert_structural_equal(example, built)

calls = []


def emit(value, *, loc=None):
    calls.append(type(value).__name__)
    return B.emit_(value, loc=loc)


M = SimpleNamespace(**vars(B))
M.emit_ = emit
M.function = make_decorator(M, namespace_path="custom.function")
register_namespace("custom", M)


@M.function
def customized(a: M.int32):
    M.evaluate(a + 1)


assert len(calls) == 1, calls
assert isinstance(customized, type(example))
print(
    json.dumps(
        {
            "before_after_structural_equal": True,
            "custom_emit_calls": calls,
            "script": str(example),
        }
    )
)
