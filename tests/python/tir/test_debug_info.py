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
"""Test line-level debug info for TIR"""
import tvm
import tvm.testing
from tvm import tir
from tvm import relay
from tvm.script import tir as T

from typing import List, Dict
import re


def find_di_locations(source: str) -> Dict[int, int]:
    """
    Parse out DILocation references in printed LLVM IR
    """
    result = {}

    for line in source.splitlines():
        m = re.match(r"!(\d+) = !DILocation\(line: (\d+).*", line)
        if m:
            debug_id, line = m.groups()
            result[debug_id] = line

    return result


def test_llvm_ir_debug_info():
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            # We exchange data between function by handles, which are similar to pointer.
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # Create buffer from handles.
            A = T.match_buffer(a, (8,), dtype="float32")
            B = T.match_buffer(b, (8,), dtype="float32")
            for i in range(8):
                # A block is an abstraction for computation.
                with T.block("B"):
                    # Define a spatial block iterator and bind it to value i.
                    vi = T.axis.spatial(8, i)
                    assert 1 == 0, "Some numbers"
                    B[vi] = A[vi] + 1.0

    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": True}):
        runtime_module = tvm.build(MyModule, target="llvm")

    source = runtime_module.get_source()

    locations = find_di_locations(source)
    assert len(locations) == 33


if __name__ == "__main__":
    tvm.testing.main()
