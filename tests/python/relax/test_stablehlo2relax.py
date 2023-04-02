# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple smoketest for the Python API."""

# pylint: disable=wildcard-import,undefined-variable
from typing import List, Any

# mlir and stablehlo from jaxlib
from jaxlib import mlir
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import *

# import relax
from tvm import relax, tir
from tvm.script.parser import relax as R


def get_shape(ipt: mlir.ir.Type) -> List[Any]:
    """Parse the Type like tensor<?x?xf32> to List
    -1 is reserved for dynamic dimension
    """
    # Get ShapeType
    shape_type = mlir.ir.ShapedType(ipt)
    ret = []
    for i in range(shape_type.rank):
        # get_dim_size
        if shape_type.is_dynamic_dim(i):
            n = tir.Var("n", "int64")
            ret.append(n)
        else:
            ret.append(shape_type.get_dim_size(i))
    return ret


def test_add():
    ASM = """
    func.func @test(%arg0: tensor<3x?xf32>, %arg1: tensor<3x?xf32>) -> tensor<3x?xf32> {
      %1 = stablehlo.add %arg0, %arg1 : (tensor<3x?xf32>, tensor<3x?xf32>) -> tensor<3x?xf32>
      func.return %1 : tensor<3x?xf32>
    }
    """

    ASM_Add_from_jax = """
    module @jit_f {
      func.func public @main(%arg0: tensor<f32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<f32> {jax.arg_info = "y", mhlo.sharding = "{replicated}"}) -> (tensor<f32> {jax.result_info = ""}) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
        return %0 : tensor<f32>
      }
    }
    """

    with Context() as context:
        stablehlo.register_dialect(context)

        m = Module.parse(ASM)  # mlir._mlir_libs._mlir.ir.Module
        print("The parsed Module: {} \n----------\nmodule type: {}".format(m, type(m)))
        assert isinstance(m, Module)
        assert isinstance(m.body, Block)
        assert isinstance(m.body.arguments, BlockArgumentList)
        assert isinstance(m.body.operations, OperationList)
        assert isinstance(m.body.operations[0].arguments, BlockArgumentList)
        assert isinstance(m.body.operations[0].regions, RegionSequence)
        assert isinstance(m.body.operations[0].regions[0], Region)
        assert isinstance(m.body.operations[0].regions[0].blocks, BlockList)
        assert isinstance(m.body.operations[0].regions[0].blocks[0], Block)
        # Looks owner and region are not necessary for translation, duplicate info.
        block = m.body.operations[0].regions[0].blocks[0]
        for op in block.operations:
            print("op: {} - type: {}".format(op, type(op)))
        assert isinstance(block.operations[0], mlir.dialects._stablehlo_ops_gen.AddOp)
        assert isinstance(block.operations[1], mlir.dialects._func_ops_gen.ReturnOp)
        add_op, return_op = block.operations
        assert isinstance(add_op.lhs, Value)
        assert isinstance(add_op.operands, mlir._mlir_libs._mlir.ir.OpOperandList)
        assert isinstance(add_op.operands[0], Value)
        lhs_operand, rhs_operand = add_op.operands
        result_value = add_op.results[0]

        lhs_shape = get_shape(lhs_operand.type)
        rhs_shape = get_shape(rhs_operand.type)
        x = relax.Var("x", R.Tensor(lhs_shape, "float32"))
        y = relax.Var("y", R.Tensor(rhs_shape, "float32"))
        bb = relax.BlockBuilder()
        with bb.function("foo", (x, y)):
            y = bb.emit(relax.op.add(x, x))
            bb.emit_func_output(y)

        bb.get().show()
        return bb.get()


if __name__ == "__main__":
    test_add()
