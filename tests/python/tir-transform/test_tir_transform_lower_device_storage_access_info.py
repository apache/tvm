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
import tvm.testing
from tvm.script import tir as T, ir as I


@tvm.register_global_func("tvm.info.mem.global.test_with_head_address")
def mem_info_with_head_address():
    return tvm.ir.make_node(
        "target.MemoryInfo",
        unit_bits=8,
        max_simd_bits=32,
        max_num_bits=128,
        head_address=tvm.tir.call_extern("handle", "dummy_head_address"),
    )


@tvm.register_global_func("tvm.info.mem.global.test_without_head_address")
def mem_info_without_head_address():
    return tvm.ir.make_node(
        "target.MemoryInfo",
        unit_bits=8,
        max_simd_bits=32,
        max_num_bits=128,
        head_address=None,
    )


def test_lower_cpu_accessible_scope():
    """Allocate of CPU-visible buffers are replaced by LetStmt

    For scopes that are accessible by the CPU (e.g. VTCM on hexagon),
    the head address specifies how it should be accessed, and is used
    to replace the AllocateNode.
    """
    transform = tvm.tir.transform.LowerDeviceStorageAccessInfo()

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            ptr = T.allocate([16], "float32", scope="global.test_with_head_address")
            T.evaluate(ptr)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            ptr: T.handle("float32", "global.test_with_head_address") = T.call_extern(  # noqa: F722
                "handle", "dummy_head_address"
            )
            T.evaluate(ptr)

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_lower_cpu_accessible_scope_with_decl_buffer():
    """Like test_lower_cpu_accessible_scope, but with a DeclBuffer.

    When the Allocate is updated, the DeclBuffer should not contain a
    dangling reference.
    """
    transform = tvm.tir.transform.LowerDeviceStorageAccessInfo()

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            buf = T.decl_buffer(16, "float32", scope="global.test_with_head_address")
            T.evaluate(buf.data)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            ptr: T.handle("float32", "global.test_with_head_address") = T.call_extern(  # noqa: F722
                "handle", "dummy_head_address"
            )
            buf = T.decl_buffer(16, "float32", scope="global.test_with_head_address", data=ptr)
            T.evaluate(ptr)

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_lower_cpu_inaccessible_scope():
    """Allocate of CPU-visible buffers are replaced by LetStmt

    For scopes that are inaccessible by the CPU (e.g. Texture memory
    on GPU), the allocate is removed.  All CPU-side references to the
    buffer should have been lowered by this point.
    """
    transform = tvm.tir.transform.LowerDeviceStorageAccessInfo()

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            ptr = T.allocate([16], "float32", scope="global.test_without_head_address")
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            T.evaluate(0)

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_lower_cpu_inaccessible_scope_with_decl_buffer():
    """Like test_lower_cpu_inaccessible_scope, but with a DeclBuffer

    When the Allocate is removed, the DeclBuffer should not contain a
    dangling reference.
    """
    transform = tvm.tir.transform.LowerDeviceStorageAccessInfo()

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            buf = T.decl_buffer(16, "float32", scope="global.test_without_head_address")
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            T.evaluate(0)

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
