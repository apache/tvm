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
# under the License
"""Unit tests for the OutlineCompilerFunctionsWithExistingGlobalSymbols and
   MarkCompilerFunctionsAsExtern external codegen helper passes."""

import tvm
import tvm.testing
import numpy as np


def make_const(dtype, shape):
    return tvm.relay.const(np.random.rand(*shape).astype(dtype))


def make_consts(dtype, shapes):
    return [make_const(dtype, shape) for shape in shapes]


metatable = {
    "relay.Constant": make_consts(
        "float16",
        [
            (2304, 768),  # 0
            (2304,),  # 1
            (600, 32, 64),  # 2
        ],
    )
}


def original_mod():
    return tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0 : Tensor[(1600, 768), float16], %x3 : Tensor[(600, 32, 64), float16]) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
          %0 = fn(%y_0_i0: Tensor[(1600, 768), float16], %y_0_i1: Tensor[(2304, 768), float16], %y_0_i2: Tensor[(2304), float16],
                  Inline=1, Compiler="cutlass", global_symbol="tvmgen_default_cutlass_main_0", Primitive=1) -> Tensor[(1600, 2304), float16] {
            %4 = fn (%FunctionVar_0_0: Tensor[(1600, 768), float16], %FunctionVar_0_1: Tensor[(2304, 768), float16], %FunctionVar_0_2: Tensor[(2304), float16],
                     PartitionedFromPattern="nn.dense_add_", Composite="cutlass.dense_bias") -> Tensor[(1600, 2304), float16] {
              %5 = nn.dense(%FunctionVar_0_0, %FunctionVar_0_1, units=2304);
              add(%5, %FunctionVar_0_2)
            };
            %4(%y_0_i0, %y_0_i1, %y_0_i2)
          };
          %1 = %0(%x0, meta[relay.Constant][0], meta[relay.Constant][1]);
          %2 = fn(%y_3_i0: Tensor[(600, 32, 64), float16], %y_3_i1: Tensor[(600, 32, 64), float16],
                  Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
            %6 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16], %FunctionVar_0_11: Tensor[(600, 32, 64), float16],
                     PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
              nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True)
            };
            %6(%y_3_i0, %y_3_i1)
          };
          %3 = %2(%x3, meta[relay.Constant][2]);
          (%1, %3)
        }
        """,
        "from_string",
        None,
        metatable,
    )


def original_mod_let_bound():
    return tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0 : Tensor[(1600, 768), float16], %x3 : Tensor[(600, 32, 64), float16]) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
          let %f = fn(%y_0_i0: Tensor[(1600, 768), float16], %y_0_i1: Tensor[(2304, 768), float16], %y_0_i2: Tensor[(2304), float16],
                      Inline=1, Compiler="cutlass", global_symbol="tvmgen_default_cutlass_main_0", Primitive=1) -> Tensor[(1600, 2304), float16] {
            %4 = fn (%FunctionVar_0_0: Tensor[(1600, 768), float16], %FunctionVar_0_1: Tensor[(2304, 768), float16], %FunctionVar_0_2: Tensor[(2304), float16],
                     PartitionedFromPattern="nn.dense_add_", Composite="cutlass.dense_bias") -> Tensor[(1600, 2304), float16] {
              %5 = nn.dense(%FunctionVar_0_0, %FunctionVar_0_1, units=2304);
              add(%5, %FunctionVar_0_2)
            };
            %4(%y_0_i0, %y_0_i1, %y_0_i2)
          };
          %1 = %f(%x0, meta[relay.Constant][0], meta[relay.Constant][1]);
          %2 = fn(%y_3_i0: Tensor[(600, 32, 64), float16], %y_3_i1: Tensor[(600, 32, 64), float16],
                  Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
            %6 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16], %FunctionVar_0_11: Tensor[(600, 32, 64), float16],
                     PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
              nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True)
            };
            %6(%y_3_i0, %y_3_i1)
          };
          %3 = %2(%x3, meta[relay.Constant][2]);
          (%1, %3)
        }
        """,
        "from_string",
        None,
        metatable,
    )


def expected_outlined_mod():
    return tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0 : Tensor[(1600, 768), float16], %x3 : Tensor[(600, 32, 64), float16]) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
          %1 = @tvmgen_default_cutlass_main_0(%x0, meta[relay.Constant][0], meta[relay.Constant][1]);
          %2 = fn(%y_3_i0: Tensor[(600, 32, 64), float16], %y_3_i1: Tensor[(600, 32, 64), float16],
                  Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
            %6 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16], %FunctionVar_0_11: Tensor[(600, 32, 64), float16],
                     PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
              nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True)
            };
            %6(%y_3_i0, %y_3_i1)
          };
          %3 = %2(%x3, meta[relay.Constant][2]);
          (%1, %3)
        }

        def @tvmgen_default_cutlass_main_0(%y_0_i0: Tensor[(1600, 768), float16], %y_0_i1: Tensor[(2304, 768), float16], %y_0_i2: Tensor[(2304), float16],
                  Inline=1, Compiler="cutlass", global_symbol="tvmgen_default_cutlass_main_0", Primitive=1) -> Tensor[(1600, 2304), float16] {
          %4 = fn (%FunctionVar_0_0: Tensor[(1600, 768), float16], %FunctionVar_0_1: Tensor[(2304, 768), float16], %FunctionVar_0_2: Tensor[(2304), float16],
                   PartitionedFromPattern="nn.dense_add_", Composite="cutlass.dense_bias") -> Tensor[(1600, 2304), float16] {
            %5 = nn.dense(%FunctionVar_0_0, %FunctionVar_0_1, units=2304);
            add(%5, %FunctionVar_0_2)
          };
          %4(%y_0_i0, %y_0_i1, %y_0_i2)
        }
        """,
        "from_string",
        None,
        metatable,
    )


def expected_extern_mod():
    return tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0 : Tensor[(1600, 768), float16], %x3 : Tensor[(600, 32, 64), float16]) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
          %1 = @tvmgen_default_cutlass_main_0(%x0, meta[relay.Constant][0], meta[relay.Constant][1]);
          %2 = fn(%y_3_i0: Tensor[(600, 32, 64), float16], %y_3_i1: Tensor[(600, 32, 64), float16],
                  Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
            %6 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16], %FunctionVar_0_11: Tensor[(600, 32, 64), float16],
                     PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
              nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True)
            };
            %6(%y_3_i0, %y_3_i1)
          };
          %3 = %2(%x3, meta[relay.Constant][2]);
          (%1, %3)
        }

        def @tvmgen_default_cutlass_main_0(%y_0_i0: Tensor[(1600, 768), float16], %y_0_i1: Tensor[(2304, 768), float16], %y_0_i2: Tensor[(2304), float16],
                  Extern=1) -> Tensor[(1600, 2304), float16] {
          %4 = fn (%FunctionVar_0_0: Tensor[(1600, 768), float16], %FunctionVar_0_1: Tensor[(2304, 768), float16], %FunctionVar_0_2: Tensor[(2304), float16],
                   PartitionedFromPattern="nn.dense_add_", Composite="cutlass.dense_bias") -> Tensor[(1600, 2304), float16] {
            %5 = nn.dense(%FunctionVar_0_0, %FunctionVar_0_1, units=2304);
            add(%5, %FunctionVar_0_2)
          };
          %4(%y_0_i0, %y_0_i1, %y_0_i2)
        }
        """,
        "from_string",
        None,
        metatable,
    )


def expected_inlined_mod():
    return tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0 : Tensor[(1600, 768), float16], %x3 : Tensor[(600, 32, 64), float16]) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
          %0 = nn.dense(%x0, meta[relay.Constant][0], units=2304);
          %1 = add(%0, meta[relay.Constant][1]);
          %2 = fn(%y_3_i0: Tensor[(600, 32, 64), float16], %y_3_i1: Tensor[(600, 32, 64), float16],
                  Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
            %6 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16], %FunctionVar_0_11: Tensor[(600, 32, 64), float16],
                     PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
              nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True)
            };
            %6(%y_3_i0, %y_3_i1)
          };
          %3 = %2(%x3, meta[relay.Constant][2]);
          (%1, %3)
        }
        """,
        "from_string",
        None,
        metatable,
    )


def test_outline_compiler_functions_with_existing_global_symbols():
    actual_outlined_mod = tvm.relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols(
        "cutlass"
    )(original_mod())
    tvm.ir.assert_structural_equal(actual_outlined_mod, expected_outlined_mod(), map_free_vars=True)


def test_outline_let_bound_compiler_functions_with_existing_global_symbols():
    actual_outlined_mod = tvm.relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols(
        "cutlass"
    )(original_mod_let_bound())
    tvm.ir.assert_structural_equal(actual_outlined_mod, expected_outlined_mod(), map_free_vars=True)


def test_mark_compiler_functions_as_extern():
    actual_extern_mod = tvm.relay.transform.MarkCompilerFunctionsAsExtern("cutlass")(
        expected_outlined_mod()
    )
    tvm.ir.assert_structural_equal(actual_extern_mod, expected_extern_mod(), map_free_vars=True)


def test_inline_compiler_functions():
    mod = expected_outlined_mod()
    gv = mod.get_global_var("tvmgen_default_cutlass_main_0")
    actual_inlined_mod = tvm.relay.transform.InlineCompilerFunctionsBoundTo([gv])(mod)
    tvm.ir.assert_structural_equal(actual_inlined_mod, expected_inlined_mod(), map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
