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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""StableHLO frontend of Relax."""
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from functools import reduce

from jaxlib import mlir
from jaxlib.mlir.dialects import stablehlo

import tvm
from tvm import relax, tir


class StableHLOImporter:
    """An importer from StableHLO to Relax."""

    def __init__(self) -> None:
        self._nodes: Dict[Union[str, mlir.ir.Operation], relax.Expr] = {}
        self.params: Dict[mlir.ir.RankedTensorType, relax.Expr] = {}
        self.named_modules: Dict[str, mlir.ir.Module] = None
        self.block_builder: relax.BlockBuilder = None
        self.create_convert_map()

    ########## Utilities ##########
    @staticmethod
    def _fetch_attr(model, target: str):

        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced non existing target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @staticmethod
    def _convert_data_type(input_type: mlir.ir.Type):
        """converts the data type from mlir to tvm."""
        if mlir.ir.ShapedType.isinstance(input_type):
            input_type = mlir.ir.ShapedType(input_type).element_type

        input_type = str(input_type)
        if input_type == "f16":
            return "float16"
        if input_type in ["f32", "F32Type"]:
            return "float32"
        elif input_type in ["f64", "F64Type"]:
            return "float64"
        elif input_type == "i1":
            return "bool"
        elif input_type == "i8":
            return "int8"
        elif input_type == "i16":
            return "int16"
        elif input_type == "i32":
            return "int32"
        elif input_type == "i64":
            return "int64"
        elif input_type == "ui8":
            return "uint8"
        elif input_type == "ui16":
            return "uint16"
        elif input_type == "ui32":
            return "uint32"
        elif input_type == "ui64":
            return "uint64"
        else:
            raise NotImplementedError("input_type {} is not handled yet".format(input_type))

    @staticmethod
    def _convert_stablehlo_tensor_to_relax(tensor: mlir.ir.RankedTensorType) -> relax.Var:
        tensor = tensor.detach().cpu()
        dtype = StableHLOImporter._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.numpy(), dtype)

    @staticmethod
    def shape_of(tensor):
        """Get the shape of a tensor."""

        if isinstance(tensor, relax.Expr):
            if not isinstance(tensor.struct_info, relax.TensorStructInfo):
                raise TypeError("The input Expr of shape_of should be a Tensor")
            return tensor.struct_info.shape
        elif isinstance(tensor, mlir.ir.RankedTensorType):
            return tensor.shape
        raise ValueError("Unsupported type: {}".format(type(tensor)))

    def _attr2value(self, node: mlir.ir.Attribute) -> Union[Any, List[Any]]:
        if mlir.ir.IntegerAttr.isinstance(node):
            int_attr = mlir.ir.IntegerAttr(node)
            return int_attr.value
        if mlir.ir.FloatAttr.isinstance(node):
            float_attr = mlir.ir.FloatAttr(node)
            return float_attr.value
        if mlir.ir.DenseIntElementsAttr.isinstance(node):
            dense_attr = mlir.ir.DenseIntElementsAttr(node)
        elif mlir.ir.DenseFPElementsAttr.isinstance(node):
            dense_attr = mlir.ir.DenseFPElementsAttr(node)
        else:
            raise ValueError("Unsupported Attribute type: " + str(type(node)))
        ret = []
        for val in dense_attr:
            ret.append(val)
        return ret

    def _getattr(self, node: mlir.ir.Operation) -> relax.Var:
        if isinstance(self._nodes[node.args[0]], relax.Expr):
            if node.args[1] == "dtype":
                return self._nodes[node.args[0]].struct_info.dtype
            elif node.args[1] == "shape":
                return self.shape_of(self._nodes[node.args[0]])
        return getattr(self._nodes[node.args[0]], node.args[1])

    def _getitem(self, node: mlir.ir.Operation) -> relax.Var:
        x = self._nodes[node.args[0]]
        if isinstance(x, (list, tuple, relax.ShapeExpr, relax.Tuple)):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.struct_info, relax.TupleStructInfo):
                return self.block_builder.emit(relax.TupleGetItem(x, node.args[1]))

            assert isinstance(x.struct_info, relax.TensorStructInfo)
            begin = []
            end = []
            stride = []
            axes = []
            expand_dim = []
            i = 0
            shape = self.shape_of(x)
            for index in node.args[1]:
                if isinstance(index, int):
                    begin.append(index)
                    end.append(index + 1)
                    stride.append(1)
                    axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    begin.append(0 if index.start is None else index.start)
                    end.append(shape[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(i)
                    i = i + 1
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(shape):
                begin.append(0)
                end.append(shape[i])
                stride.append(1)
                axes.append(i)
                i = i + 1
            sliced = self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))
            sliced_shape = list(self.shape_of(sliced))
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.block_builder.emit(relax.op.reshape(sliced, sliced_shape))
        elif isinstance(x, relax.Constant):
            dtype = x.struct_info.dtype
            return relax.const(x.data.numpy()[node.args[1]], dtype)
        else:
            assert False

    def retrieve_operands(self, node):
        return self._retrieve_operands(node.operands)

    def _retrieve_operands(self, node):
        # the operand is one of the inputs of FuncOp
        if isinstance(node, mlir.ir.Operation):
            return self._nodes[node]
        if isinstance(node, tuple):
            return tuple(self._retrieve_operands(x) for x in node)
        if isinstance(node, (list, mlir.ir.OpOperandList)):
            return [self._retrieve_operands(x) for x in node]
        if isinstance(node, dict):
            return {self._retrieve_operands(k): self._retrieve_operands(v) for k, v in node.items()}
        if isinstance(node, mlir.ir.Value):
            if isinstance(node.owner, mlir.ir.Block):
                block_arg = mlir.ir.BlockArgument(node)
                return self._nodes["arg" + str(block_arg.arg_number)]
            return self._retrieve_operands(node.owner)
        return node

    def get_shape(self, inpt_type: mlir.ir.ShapedType) -> List[Any]:
        """Get the shape from Type like tensor<?x?xf32>"""
        shape_type = inpt_type
        if isinstance(shape_type, mlir.ir.Type):
            shape_type = mlir.ir.ShapedType(shape_type)
        dtype = self._convert_data_type(shape_type.element_type)
        ret = []
        for i in range(shape_type.rank):
            # get_dim_size
            if shape_type.is_dynamic_dim(i):
                n = tir.Var("n", "int64")
                ret.append(n)
            else:
                ret.append(shape_type.get_dim_size(i))

        return ret

    @staticmethod
    def _promote_binary_op_args(lhs, rhs):
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return lhs, rhs
        elif isinstance(lhs, relax.Expr):
            assert isinstance(lhs.struct_info, relax.TensorStructInfo)
            return lhs, relax.const(rhs, lhs.struct_info.dtype)
        elif isinstance(rhs, relax.Expr):
            assert isinstance(rhs.struct_info, relax.TensorStructInfo)
            return relax.const(lhs, rhs.struct_info.dtype), rhs
        else:
            assert False

    def _call_binary_op(self, op, lhs, rhs):
        lhs, rhs = StableHLOImporter._promote_binary_op_args(lhs, rhs)
        return self.block_builder.emit(op(lhs, rhs))

    def _add(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.add, lhs, rhs)
        return lhs + rhs

    def _maximum(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.maximum(lhs, rhs))

    def _minimum(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.minimum(lhs, rhs))

    def _divide(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.divide, lhs, rhs)
        return lhs / rhs

    def _multiply(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.multiply, lhs, rhs)
        return lhs * rhs

    def _subtract(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.subtract, lhs, rhs)
        return lhs - rhs

    def _broadcast_in_dim(self, node: mlir.ir.Operation) -> relax.Expr:
        operands = self.retrieve_operands(node)
        data = operands[0]
        # broadcast_dims = self._attr2value(node.attributes["broadcast_dimensions"])
        shape = self.get_shape(node.result.type)
        # scalar
        if len(shape) == 0:
            return data
        return self.block_builder.emit(relax.op.broadcast_to(data, shape))

    def _const(self, node: mlir.ir.Operation) -> relax.Expr:
        const_value = self._attr2value(node.attributes["value"])
        dtype = self._convert_data_type(node.result.type)
        return relax.const(const_value, dtype)

    def _dot_general(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.matmul(lhs, rhs))

    def _convolution(self, node: mlir.ir.Operation) -> relax.Expr:
        x, weight = self.retrieve_operands(node)
        shaped_type = mlir.ir.ShapedType(node.result.type)
        out_dtype = self._convert_data_type(shaped_type.element_type)
        strides = self._attr2value(node.attributes["window_strides"])
        padding = self._attr2value(node.attributes["padding"])
        padding = self._attr2value(node.attributes["padding"])
        lhs_dilation = self._attr2value(node.attributes["lhs_dilation"])
        rhs_dilation = self._attr2value(node.attributes["rhs_dilation"])
        if len(lhs_dilation) > 0:
            lhs_dilation = lhs_dilation[0]
        if len(rhs_dilation) > 0:
            rhs_dilation = rhs_dilation[0]
        # todo (yongwww): how to get dilation if lhs and rhs is not squired
        dilation = (lhs_dilation, rhs_dilation)
        # TODO (yongwww): Remove hack for padding
        tmp = padding[1]
        padding[1] = padding[2]
        padding[2] = tmp
        # todo(yongwww): fix, feature_group_count ? batch_group_count
        groups = self._attr2value(node.attributes["batch_group_count"])
        conv2d = relax.op.nn.conv2d(
            x,
            weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype=out_dtype,
        )

        return self.block_builder.emit(conv2d)

    def _reshape(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        if isinstance(data, list):
            assert len(data) == 1
            data = data[0]
        new_shape = self.get_shape(node.result.type)
        return self.block_builder.emit(relax.op.reshape(data, new_shape))

    def _reduce(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        # TODO(YONGWWW): add other reduce type like subtract
        return self.block_builder.emit(relax.op.add(data[0], data[1]))

    def _reduce_window(self, node: mlir.ir.Operation) -> relax.Expr:
        operands = self.retrieve_operands(node)
        window_dimensions = self._attr2value(node.attributes["window_dimensions"])
        window_dilations = self._attr2value(node.attributes["window_dilations"])

        pool_size = []
        for i, window_dim in enumerate(window_dimensions):
            if window_dim == 0:
                pool_size.append(0)
            else:
                dilated_window_size = (window_dim - 1) * window_dilations[i] + 1
                pool_size.append(dilated_window_size)
        strides = self._attr2value(node.attributes["window_strides"])
        padding = self._attr2value(
            node.attributes["padding"]
        )  # todo: the padding logic might be wrong

        layout = "NHWC"  # todo (how to automatically figure it out), and func is add
        return self.block_builder.emit(
            relax.op.nn.max_pool2d(
                operands[0],
                pool_size=pool_size[1:3],  # HW
                strides=strides[1:3],
                padding=[0, 0, 1, 1],  # hack
                # dilation=window_dilations,
                layout=layout,
            )
        )

    def _rsqrt(self, node: mlir.ir.Operation) -> relax.Expr:
        # TODO (relax-team): Add rsqrt to Relax
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.power(data[0], relax.const(-0.5)))

    def _return(self, node: mlir.ir.Operation) -> relax.Expr:
        outputs = self.retrieve_operands(node)
        if isinstance(outputs, list):
            pass  # todo(yongwww)
        return self.block_builder.emit_output(self.nodes[outputs])

    def create_convert_map(self):

        self.convert_map: Dict[str, Callable[[mlir.ir.Operation], relax.Var]] = {
            "stablehlo.add": self._add,
            "stablehlo.broadcast_in_dim": self._broadcast_in_dim,
            "stablehlo.constant": self._const,
            "stablehlo.convolution": self._convolution,
            "stablehlo.divide": self._divide,
            "stablehlo.dot_general": self._dot_general,
            "stablehlo.maximum": self._maximum,
            "stablehlo.minimum": self._minimum,
            "stablehlo.multiply": self._multiply,
            "stablehlo.reshape": self._reshape,
            "stablehlo.reduce": self._reduce,
            "stablehlo.reduce_window": self._reduce_window,
            "stablehlo.rsqrt": self._rsqrt,
            "stablehlo.subtract": self._subtract,
            "func.return": self._return,
            "stablehlo.return": self._return,
            "getattr": self._getattr,
            "getitem": self._getitem,
            "contiguous": lambda node: self._nodes[node.args[0]],
        }

    def from_stablehlo(self, model, input_info: List[Tuple[Tuple[int], str]]) -> tvm.IRModule:
        """Convert a StableHLO Module to a Relax program."""
        # Nothing in model.body.arguments
        block: mlir.ir.Block = model.body.operations[0].regions[0].blocks[0]

        # inputs of the function
        inputs = []
        for idx, arg in enumerate(block.arguments.types):
            arg_shape = mlir.ir.ShapedType(arg)
            ipt_shape = self.get_shape(arg_shape)
            ipt_dtype = self._convert_data_type(arg_shape.element_type)
            ipt_name = "arg" + str(idx)
            ipt_var = relax.Var(f"arg{idx}", relax.TensorStructInfo(ipt_shape, ipt_dtype))
            self._nodes[ipt_name] = ipt_var
            inputs.append(ipt_var)

        # todo: Figure out input names
        # func_op = m.body.operations[0]
        # func_name = func_op.name
        # func_op.type: Type((tensor<3x?xf32>, tensor<3x?xf32>) -> tensor<3x?xf32>)
        # func_op.type.inputs
        # [Type(tensor<3x?xf32>), Type(tensor<3x?xf32>)]

        # TODO (yongwww): Handle mlir.ir.Module with multiple functions
        # Initialize the block builder with a function and a dataflow block.
        func_name = "main"
        self.block_builder = relax.BlockBuilder()

        with self.block_builder.function(name=func_name, params=inputs.copy()):
            output = None
            with self.block_builder.dataflow():
                block = model.body.operations[0].regions[0].blocks[0]
                tmp = 0
                for operation in block.operations:
                    if isinstance(operation, (mlir.dialects.func.ReturnOp, stablehlo.ReturnOp)):
                        operation = operation.operands[0].owner
                        # TODO (yongwww): handle multiple outputs
                        output = self.block_builder.emit_output(self._nodes[operation])
                        break

                    elif isinstance(operation, mlir.ir.OpView):
                        op_name = operation.operation.name
                        # if op_name == "stablehlo.reduce_window":
                        # if tmp == 104:
                        #    output = self.block_builder.emit_output(inputs[1])
                        #    break
                        tmp = tmp + 1

                        assert op_name in self.convert_map, f"Unsupported operation {op_name}"
                        self._nodes[operation] = self.convert_map[op_name](operation)
                    else:
                        raise ValueError(f"Unsupported op {operation}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        return mod


def from_stablehlo(
    model,
    input_info: List[Tuple[Tuple[int], str]] = None,
) -> tvm.IRModule:
    """Convert a StableHLO Module to a Relax program

    Parameters
    ----------
    model : mlir.ir.Module
        The StableHLO Module to convert.

    input_info : List[Tuple[Tuple[int], str]]
        A list of shapes and data types of input tensors.

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.
    """
    return StableHLOImporter().from_stablehlo(model, input_info)
