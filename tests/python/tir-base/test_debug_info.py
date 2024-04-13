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
from tvm.script import tir as T, ir as I

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


def _module():
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            # We exchange data between function by handles, which are similar to pointer.
            T.func_attr(
                {
                    "tir.noalias": True,
                    "target": T.target("llvm"),
                }
            )
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

    return MyModule


def test_tir_debug_info():
    """
    Test that Spans are correctly replaced with debug spans that reference
    the printed TIR
    """

    def find_span(m):
        func = next(m.functions.values())
        return func.body.block.body.span

    module_before = _module()
    span_before = find_span(module_before)
    assert span_before is None

    module_after = tir.transform.InstallDebugSpans()(module_before)
    span_after = find_span(module_after)

    # Check that the module name has been added and a line number is present
    assert span_after.source_name.name == "main.tir"
    assert span_after.line == 4


def test_tir_debug_info_with_subroutine():
    """Like test_tir_debug_info, but with a TIR subroutine

    The current InstallDebugSpans applies to a single PrimFunc.  This
    test verifies that the existence of device-side subroutines

    """

    def find_span(m):
        func = next(m.functions.values())
        return func.body.block.body.span

    @tvm.script.ir_module
    class module_before:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True, "target": T.target("llvm")})
            A = T.match_buffer(a, (8,), dtype="float32")
            B = T.match_buffer(b, (8,), dtype="float32")
            for i in range(8):
                with T.block("B"):
                    vi = T.axis.spatial(8, i)
                    module_before.subroutine(T.address_of(A[vi]), T.address_of(B[vi]))

        @T.prim_func
        def subroutine(a_ptr: T.handle("float32"), b_ptr: T.handle("float32")):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.decl_buffer(1, "float32", data=a_ptr)
            B = T.decl_buffer(1, "float32", data=b_ptr)
            B[0] = A[1] + 1.0

    span_before = find_span(module_before)
    assert span_before is None

    module_after = tir.transform.InstallDebugSpans()(module_before)
    span_after = find_span(module_after)

    # Check that the module name has been added and a line number is present
    assert span_after.source_name.name == "main.tir"
    assert span_after.line == 4


def test_llvm_ir_debug_info():
    """
    Check that the right amount of debug locations are present
    """
    MyModule = _module()
    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": True}):
        runtime_module = tvm.build(MyModule, target="llvm")

    source = runtime_module.get_source()

    locations = find_di_locations(source)
    assert len(locations) == 41


def test_llvm_ir_debug_accuracy():
    """
    Check that the debug location on an assert is correct
    """
    MyModule = _module()
    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": True}):
        runtime_module = tvm.build(MyModule, target="llvm")
    source = runtime_module.get_source()
    locations = find_di_locations(source)

    # Find the 'assert' from MyModule
    debug_dir_match = re.search(r"tail call void %0\(.* !dbg !(\d+)\n", source)

    # Extract out the debug directive line
    directive_idx = debug_dir_match.groups()[0]

    # Check that it matches the expected line number (in main.tir)
    debug_line_no = int(locations[directive_idx])
    assert debug_line_no == 60


def test_building_without_llvm_equivalent():
    """A TIR PrimFunc may contain non-LLVM types

    Types used in optimized kernels (e.g. "e4m3_float8") may not have
    an equivalent in DWARF, or the mapping from TIR type to DWARF type
    may not be defined.  If this occurs, the function should still be
    able to be built.
    """

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(A_data: T.handle("e4m3_float8"), B_data: T.handle("e4m3_float8")):
            A = T.decl_buffer(128, "e4m3_float8", data=A_data)
            B = T.decl_buffer(128, "e4m3_float8", data=B_data)
            for i in range(128):
                B[i] = A[i]

    tvm.target.codegen.build_module(Module, "llvm")


if __name__ == "__main__":
    tvm.testing.main()
