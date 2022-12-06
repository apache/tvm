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

"""
Testing the pass that moves allocate nodes to the body of the function.
"""
# pylint: disable=wrong-import-position

import pytest

pytest.importorskip("ethosu.vela")

import tvm
from tvm.script import tir as T
from tvm.relay.backend.contrib.ethosu.tir.passes import HoistAllocates


class ExtractAllocateInfo:
    """
    Extracts information from allocate nodes which we will use as sanity to check the allocate
    after mutation.
    """

    def __init__(self):
        self.allocates_info = []

    def __call__(self, mod):
        tvm.tir.stmt_functor.ir_transform(mod["main"].body, self._pre_visit, None, ["tir.Allocate"])
        return self.allocates_info

    def _pre_visit(self, stmt):
        self.allocates_info.append(
            {"extents": stmt.extents, "dtype": stmt.dtype, "condition": stmt.condition}
        )


def CheckAllocates(allocate_info):  # pylint: disable=invalid-name
    """
    Checks that all allocates have been visited before an external call has been visited and
    checks that the information for each allocate is what is expected. Additionally, the pass
    checks the body of the tir after the final allocate statement is flat (it contains no
    sequence statement).
    """

    allocate_idx = 0
    expected_num_allocates = len(allocate_info)
    num_seq_stmts = 0

    def _pre_visit(stmt):
        nonlocal allocate_idx, expected_num_allocates, num_seq_stmts

        if isinstance(stmt, tvm.tir.Allocate):
            expected = allocate_info[allocate_idx]
            assert (
                stmt.extents == expected["extents"]
            ), f"Allocate extents {stmt.extents} did not match expected {expected['extents']}"
            assert (
                stmt.dtype == expected["dtype"]
            ), f"Allocate dtype {stmt.dtype} did not match expected {expected['dtype']}"
            assert (
                stmt.condition == expected["condition"]
            ), f"Allocate condition {stmt.condition} did not match expected {expected['condition']}"

            allocate_idx += 1
        elif isinstance(stmt, tvm.tir.SeqStmt):
            num_seq_stmts += 1
            assert num_seq_stmts <= expected_num_allocates, (
                "Encountered a SeqStmt after all allocates have been visited, was the "
                "body flattened correctly?"
            )
        else:
            assert (
                allocate_idx == expected_num_allocates
            ), "A call node was visited before all allocates"

    def _ftransform(f, mod, ctx):
        f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body, _pre_visit, None, ["tir.Allocate", "tir.Call", "tir.SeqStmt"]
            )
        )

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)


def test_double_convolution():
    """
    Test to check the HoistAllocates pass works on a function with two convolutions.
    """

    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1, 27, 42, 3), "int8"], input_placeholder_encoded: T.Buffer[(3, 3, 2, 3), "uint8"], input_placeholder_encoded_1: T.Buffer[(3, 10), "uint8"], input_placeholder_encoded_2: T.Buffer[(3, 3, 2, 3), "uint8"], input_placeholder_encoded_3: T.Buffer[(3, 10), "uint8"], input_ethosu_write: T.Buffer[(1, 27, 42, 3), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            placeholder = T.buffer_decl([3402], dtype="int8", data=input_placeholder.data)
            placeholder_encoded = T.buffer_decl([128], dtype="int8", data=input_placeholder_encoded.data)
            placeholder_encoded_1 = T.buffer_decl([32], dtype="uint8", data=input_placeholder_encoded_1.data)
            placeholder_encoded_2 = T.buffer_decl([128], dtype="int8", data=input_placeholder_encoded_2.data)
            placeholder_encoded_3 = T.buffer_decl([32], dtype="uint8", data=input_placeholder_encoded_3.data)
            ethosu_write = T.buffer_decl([3402], dtype="int8", data=input_ethosu_write.data)
            # body
            placeholder_global_data = T.allocate([128], "uint8", "global")
            placeholder_global = T.buffer_decl([128], "uint8", data=placeholder_global_data)
            T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded[0], 128, placeholder_global[0], dtype="handle"))
            placeholder_d_global_data = T.allocate([32], "uint8", "global")
            placeholder_d_global = T.buffer_decl([32], "uint8", data=placeholder_d_global_data)
            T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_1[0], 32, placeholder_d_global[0], dtype="handle"))
            ethosu_write_2_data = T.allocate([18144], "int8", "global")
            ethosu_write_2 = T.buffer_decl([18144], "int8", data=ethosu_write_2_data)
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 27, 42, 3, 27, 0, 42, placeholder[0], 0, 0, 0, T.float32(0.0039215646684169769), -128, "NHWC", 126, 3, 1, "int8", 27, 42, 3, 27, 0, 42, ethosu_write_2[0], 0, 0, 0, T.float32(0.031308155506849289), -128, "NHCWB16", 672, 16, 1, 2, 3, 1, 1, 1, 2, placeholder_global[0], 128, 0, placeholder_d_global[0], 32, 2, 0, 2, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
            placeholder_d_global_1_data = T.allocate([128], "uint8", "global")
            placeholder_d_global_1 = T.buffer_decl([128], "uint8", data=placeholder_d_global_1_data)
            T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_2[0], 128, placeholder_d_global_1[0], dtype="handle"))
            placeholder_d_global_2_data = T.allocate([32], "uint8", "global")
            placeholder_d_global_2 = T.buffer_decl([32], "uint8", data=placeholder_d_global_2_data)
            T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_3[0], 32, placeholder_d_global_2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 27, 42, 3, 27, 0, 42, ethosu_write_2[0], 0, 0, 0, T.float32(0.031308155506849289), -128, "NHCWB16", 672, 16, 1, "int8", 27, 42, 3, 27, 0, 42, ethosu_write[0], 0, 0, 0, T.float32(0.23604340851306915), -128, "NHWC", 126, 3, 1, 2, 3, 1, 1, 1, 2, placeholder_d_global_1[0], 128, 0, placeholder_d_global_2[0], 32, 2, 0, 2, 1, "CLIP", -128, 127, "TFL", "NONE", dtype="handle"))
    # fmt: on

    mod = Module
    allocate_info = ExtractAllocateInfo()(mod)
    mod = HoistAllocates()(mod)
    CheckAllocates(allocate_info)(mod)


def test_identities():
    """
    Test to check the HoistAllocates pass works on a function with multiple identity
    operations, with no copy operations.
    """

    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1, 2, 3, 4), "int8"], T_concat: T.Buffer[(24,), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            placeholder = T.buffer_decl([24], dtype="int8", data=input_placeholder.data)
            # body
            ethosu_write_data = T.allocate([12], "int8", "global")
            ethosu_write = T.buffer_decl([12], "int8", data=ethosu_write_data)
            T.evaluate(T.call_extern("ethosu_identity", "int8", 1, 3, 4, 1, 0, 3, placeholder[12], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "int8", 1, 3, 4, 1, 0, 3, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
            ethosu_write_1_data = T.allocate([12], "int8", "global")
            ethosu_write_1 = T.buffer_decl([12], "int8", data=ethosu_write_1_data)
            T.evaluate(T.call_extern("ethosu_identity", "int8", 1, 3, 4, 1, 0, 3, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "int8", 1, 3, 4, 1, 0, 3, ethosu_write_1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
            T.evaluate(T.call_extern("ethosu_identity", "int8", 12, 1, 1, 12, 0, 1, ethosu_write_1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 1, 1, "int8", 12, 1, 1, 12, 0, 1, T_concat[12], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 1, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
            ethosu_write_2_data = T.allocate([12], "int8", "global")
            ethosu_write_2 = T.buffer_decl([12], "int8", data=ethosu_write_2_data)
            T.evaluate(T.call_extern("ethosu_identity", "int8", 1, 3, 4, 1, 0, 3, placeholder[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "int8", 1, 3, 4, 1, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
            ethosu_write_3_data = T.allocate([12], "int8", "global")
            ethosu_write_3 = T.buffer_decl([12], "int8", data=ethosu_write_3_data)
            T.evaluate(T.call_extern("ethosu_identity", "int8", 1, 3, 4, 1, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "int8", 1, 3, 4, 1, 0, 3, ethosu_write_3[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
            T.evaluate(T.call_extern("ethosu_identity", "int8", 12, 1, 1, 12, 0, 1, ethosu_write_3[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 1, 1, "int8", 12, 1, 1, 12, 0, 1, T_concat[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 1, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    # fmt: on

    mod = Module
    allocate_info = ExtractAllocateInfo()(mod)
    mod = HoistAllocates()(mod)
    CheckAllocates(allocate_info)(mod)


def test_outer_seq_stmt():
    """
    Test to check the HoistAllocates pass works on a function where the outer-most statement is
    a sequence statement, rather than the usual allocate.
    """

    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1, 16, 16, 32), "int8"], input_ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"], buffer_encoded: T.Buffer[(128,), "uint8"], buffer_encoded_1: T.Buffer[(32,), "uint8"], buffer_encoded_2: T.Buffer[(112,), "uint8"], buffer_encoded_3: T.Buffer[(32,), "uint8"], buffer_encoded_4: T.Buffer[(112,), "uint8"], buffer_encoded_5: T.Buffer[(32,), "uint8"], buffer_encoded_6: T.Buffer[(112,), "uint8"], buffer_encoded_7: T.Buffer[(32,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            placeholder = T.buffer_decl([8192], dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl([2048], dtype="int8", data=input_ethosu_write.data)
            # body
            with T.allocate([128], "uint8", "global") as placeholder_global_data:
                placeholder_global = T.buffer_decl([128], "uint8", data=placeholder_global_data)
                T.evaluate(T.call_extern("ethosu_copy", buffer_encoded[0], 128, placeholder_global[0], dtype="handle"))
                placeholder_d_global_data = T.allocate([32], "uint8", "global")
                placeholder_d_global = T.buffer_decl([32], "uint8", data=placeholder_d_global_data)
                T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_1[0], 32, placeholder_d_global[0], dtype="handle"))
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 128, 12, placeholder_d_global[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.allocate([112], "uint8", "global") as placeholder_global_1_data:
                placeholder_global_1 = T.buffer_decl([112], "uint8", data=placeholder_global_1_data)
                T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_2[0], 112, placeholder_global_1[0], dtype="handle"))
                placeholder_d_global_1_data = T.allocate([32], "uint8", "global")
                placeholder_d_global_1 = T.buffer_decl([32], "uint8", data=placeholder_d_global_1_data)
                T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_3[0], 32, placeholder_d_global_1[0], dtype="handle"))
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_1[0], 112, 12, placeholder_d_global_1[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.allocate([112], "uint8", "global") as placeholder_global_2_data:
                placeholder_global_2 = T.buffer_decl([112], "uint8", data=placeholder_global_2_data)
                T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_4[0], 112, placeholder_global_2[0], dtype="handle"))
                placeholder_d_global_2_data = T.allocate([32], "uint8", "global")
                placeholder_d_global_2 = T.buffer_decl([32], "uint8", data=placeholder_d_global_2_data)
                T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_5[0], 32, placeholder_d_global_2[0], dtype="handle"))
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_2[0], 112, 12, placeholder_d_global_2[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            placeholder_global_3_data = T.allocate([112], "uint8", "global")
            placeholder_global_3 = T.buffer_decl([112], "uint8", data=placeholder_global_3_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_6[0], 112, placeholder_global_3[0], dtype="handle"))
            placeholder_d_global_3_data = T.allocate([32], "uint8", "global")
            placeholder_d_global_3 = T.buffer_decl([32], "uint8", data=placeholder_d_global_3_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_7[0], 32, placeholder_d_global_3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_3[0], 112, 12, placeholder_d_global_3[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    mod = Module
    allocate_info = ExtractAllocateInfo()(mod)
    mod = HoistAllocates()(mod)
    CheckAllocates(allocate_info)(mod)


def test_allocate_without_seq_stmt():
    """
    Tests the case when an allocate statement does not have a sequence statement as its body.
    """
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1, 16, 16, 32), "int8"], input_ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"], buffer_encoded: T.Buffer[(128,), "uint8"], buffer_encoded_1: T.Buffer[(32,), "uint8"], buffer_encoded_2: T.Buffer[(112,), "uint8"], buffer_encoded_3: T.Buffer[(32,), "uint8"], buffer_encoded_4: T.Buffer[(112,), "uint8"], buffer_encoded_5: T.Buffer[(32,), "uint8"], buffer_encoded_6: T.Buffer[(112,), "uint8"], buffer_encoded_7: T.Buffer[(32,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            placeholder = T.buffer_decl([8192], dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl([2048], dtype="int8", data=input_ethosu_write.data)
            # body
            placeholder_global_data = T.allocate([128], "uint8", "global")
            placeholder_global = T.buffer_decl([128], "uint8", data=placeholder_global_data)
            placeholder_global_1_data = T.allocate([112], "uint8", "global")
            placeholder_global_1 = T.buffer_decl([112], "uint8", data=placeholder_global_1_data)
            placeholder_global_2_data = T.allocate([112], "uint8", "global")
            placeholder_global_2 = T.buffer_decl([112], "uint8", data=placeholder_global_2_data)
            placeholder_d_global_data = T.allocate([32], "uint8", "global")
            placeholder_d_global = T.buffer_decl([32], "uint8", data=placeholder_d_global_data)
            placeholder_d_global_1_data = T.allocate([32], "uint8", "global")
            placeholder_d_global_1 = T.buffer_decl([32], "uint8", data=placeholder_d_global_1_data)
            placeholder_d_global_2_data = T.allocate([32], "uint8", "global")
            placeholder_d_global_2 = T.buffer_decl([32], "uint8", data=placeholder_d_global_2_data)
            placeholder_global_3_data = T.allocate([112], "uint8", "global")
            placeholder_global_3 = T.buffer_decl([112], "uint8", data=placeholder_global_3_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded[0], 128, placeholder_global[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_1[0], 32, placeholder_d_global[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 128, 12, placeholder_d_global[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_2[0], 112, placeholder_global_1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_3[0], 32, placeholder_d_global_1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_1[0], 112, 12, placeholder_d_global_1[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_4[0], 112, placeholder_global_2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_5[0], 32, placeholder_d_global_2[0], dtype="handle"))
            placeholder_d_global_3_data = T.allocate([32], "uint8", "global")
            placeholder_d_global_3 = T.buffer_decl([32], "uint8", data=placeholder_d_global_3_data)
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_2[0], 112, 12, placeholder_d_global_2[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_6[0], 112, placeholder_global_3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_7[0], 32, placeholder_d_global_3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_3[0], 112, 12, placeholder_d_global_3[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    mod = Module
    allocate_info = ExtractAllocateInfo()(mod)
    mod = HoistAllocates()(mod)
    CheckAllocates(allocate_info)(mod)


def test_multiple_prim_funcs():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main():
            T.evaluate(0)

        @T.prim_func
        def abc():
            T.evaluate(0)

    mod = Module

    err_rgx = (
        r"Expected a single primitive function called 'main'. "
        r"Please run the HoistAllocates pass in conjunction with the LowerToTIR\(\) pass."
    )
    with pytest.raises(tvm.TVMError, match=err_rgx):
        mod = HoistAllocates()(mod)


def test_no_main_prim_func():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def abs():
            T.evaluate(0)

    mod = Module

    err_rgx = (
        r"Expected a single primitive function called 'main'. "
        r"Please run the HoistAllocates pass in conjunction with the LowerToTIR\(\) pass."
    )
    with pytest.raises(tvm.TVMError, match=err_rgx):
        mod = HoistAllocates()(mod)
