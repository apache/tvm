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

"""Implementation of TIR operator."""

from tvm.ir import Op
from tvm.tirx import Expr
from tvm.tirx.stmt import TilePrimitiveCall


def get_tirx_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirx.tile." + op_name)


class ArgProperty:
    def __init__(self, index):
        self.index = index

    def __get__(self, obj, objtype=None):
        assert obj is not None, "TilePrimitiveCall cannot be None"
        return obj.args[self.index]


### Base Operator Classes ###
class UnaryOp(TilePrimitiveCall):
    """Base class for unary operators: unary(output, input).

    Unary operators take a single input tensor and produce a single output tensor.
    """

    scalar_input = False
    output = ArgProperty(0)
    input = ArgProperty(1)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expression (input) of the operator."""
        return [self.input]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]


class UnaryOpWithBiasScale(UnaryOp):
    """Extended unary operator with bias and scale parameters: unary_with_bias_scale(output, input, bias, scale).

    These operators support additional bias and scale parameters for more complex operations (only on trn).
    output = unary(input * scale + bias)
    """  # noqa: E501

    bias = ArgProperty(2)
    scale = ArgProperty(3)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.input, self.bias, self.scale]


class BinaryOp(TilePrimitiveCall):
    """Base class for binary operators: binary(output, input0, input1).

    Binary operators take two input tensors and produce a single output tensor.
    """

    lhs = ArgProperty(1)
    rhs = ArgProperty(2)
    output = ArgProperty(0)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.lhs, self.rhs]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]


class ReduceOp(TilePrimitiveCall):
    """Base class for reduction operators: reduce(output, input, reduce_axes, accum).

    Reduction operators reduce one or more dimensions of the input tensor.
    """

    input = ArgProperty(1)
    output = ArgProperty(0)
    reduce_axes = ArgProperty(2)
    accum = ArgProperty(3)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expression (input) of the operator."""
        return [self.input]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]


### Schedule Operators ###
class Zero(UnaryOp):
    """Zero out all elements in src and store to dst."""

    op = get_tirx_op("zero")


class Sqrt(UnaryOpWithBiasScale):
    """Compute square root of all elements in src and store to dst.

    If bias and scale are provided: dst = sqrt(src * scale + bias)
    """

    op = get_tirx_op("sqrt")


class Fill(UnaryOp):
    """Fill dst with a scalar value."""

    op = get_tirx_op("fill")
    scalar_input = True


class Add(BinaryOp):
    """Add src1 and src2 element-wise and store to dst."""

    op = get_tirx_op("add")


class Sub(BinaryOp):
    """Subtract src2 from src1 element-wise and store to dst."""

    op = get_tirx_op("sub")


class Mul(BinaryOp):
    """Multiply src1 and src2 element-wise and store to dst."""

    op = get_tirx_op("mul")


class FDiv(BinaryOp):
    """Divide src1 by src2 element-wise using floating point division and store to dst."""

    op = get_tirx_op("fdiv")


class FMA(TilePrimitiveCall):
    """Fused multiply-add: output = input * scale + bias.

    fma(output, input, scale, bias)

    scale and bias can each be either a BufferRegion or a Expr scalar.
    """

    op = get_tirx_op("fma")

    output = ArgProperty(0)
    input = ArgProperty(1)
    scale = ArgProperty(2)
    bias = ArgProperty(3)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.input, self.scale, self.bias]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]


class Cast(UnaryOp):
    """Cast src to dst."""

    op = get_tirx_op("cast")


class Copy(TilePrimitiveCall):
    """Copy all elements from src to dst.

    Args:
        dst: Destination buffer region
        src: Source buffer region
    """

    op = get_tirx_op("copy")

    dst = ArgProperty(0)
    src = ArgProperty(1)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.src]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.dst]


class CopyAsync(TilePrimitiveCall):
    """Copy all elements from src to dst asynchronously.

    Args:
        dst: Destination buffer region
        src: Source buffer region
    """

    op = get_tirx_op("copy_async")

    dst = ArgProperty(0)
    src = ArgProperty(1)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.src]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.dst]


class Gemm(TilePrimitiveCall):
    """General matrix multiplication: D = A * B * alpha + C * beta.

    Args:
        D: Output matrix
        A: First input matrix
        B: Second input matrix
        C: Third input matrix (for bias)
        transpose_A: Whether to transpose A
        transpose_B: Whether to transpose B
        alpha: Scalar multiplier for A*B
        beta: Scalar multiplier for C
    """

    op = get_tirx_op("gemm")
    output = ArgProperty(0)
    lhs = ArgProperty(1)
    rhs = ArgProperty(2)
    bias = ArgProperty(3)
    transpose_A = ArgProperty(4)
    transpose_B = ArgProperty(5)
    alpha = ArgProperty(6)
    beta = ArgProperty(7)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source matrices."""
        return [self.lhs, self.rhs, self.bias]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination matrix."""
        return [self.output]


class GemmAsync(TilePrimitiveCall):
    """General matrix multiplication asynchronously.

    Supports two arg layouts:
    - Regular (6 args): C, A, B, transA, transB, accum
    - Block-scaled (8 args): C, A, B, SFA, SFB, transA, transB, accum
    """

    op = get_tirx_op("gemm_async")
    output = ArgProperty(0)
    lhs = ArgProperty(1)
    rhs = ArgProperty(2)

    @property
    def is_block_scaled(self) -> bool:
        """Whether this is a block-scaled MMA operation."""
        return len(self.args) == 8

    @property
    def sfa(self):
        """Get the scale factor buffer for A (None for regular MMA)."""
        return self.args[3] if self.is_block_scaled else None

    @property
    def sfb(self):
        """Get the scale factor buffer for B (None for regular MMA)."""
        return self.args[4] if self.is_block_scaled else None

    @property
    def transA(self):
        return self.args[5] if self.is_block_scaled else self.args[3]

    @property
    def transB(self):
        return self.args[6] if self.is_block_scaled else self.args[4]

    @property
    def accum(self):
        return self.args[7] if self.is_block_scaled else self.args[5]

    @property
    def srcs(self) -> list[Expr]:
        """Get the source matrices (including scale factors if block-scaled)."""
        srcs = [self.lhs, self.rhs]
        if self.is_block_scaled:
            srcs.extend([self.sfa, self.sfb])
        return srcs

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination matrix."""
        return [self.output]


class Sum(ReduceOp):
    """Sum elements in src along specified axes and store in dst."""

    op = get_tirx_op("sum")


class Max(ReduceOp):
    """Compute maximum value in src along specified axes and store in dst."""

    op = get_tirx_op("max")


class Min(ReduceOp):
    """Compute minimum value in src along specified axes and store in dst."""

    op = get_tirx_op("min")


class Reciprocal(UnaryOp):
    """Compute reciprocal (1/x) for all elements in src and store to dst."""

    op = get_tirx_op("reciprocal")


class SiLU(UnaryOp):
    """Compute SiLU (x * sigmoid(x)) for all elements in src and store to dst."""

    op = get_tirx_op("silu")


class Memset(UnaryOp):
    """Set all elements in dst to a specified value."""

    op = get_tirx_op("memset")
    scalar_input = True


class Maximum(BinaryOp):
    """Compute element-wise maximum of src1 and src2 and store to dst."""

    op = get_tirx_op("maximum")


class Minimum(BinaryOp):
    """Compute element-wise minimum of src1 and src2 and store to dst."""

    op = get_tirx_op("minimum")


class Exp(UnaryOpWithBiasScale):
    """Compute exponential (e^x) of all elements in src and store to dst.

    If bias and scale are provided: dst = exp(src * scale + bias)
    """

    op = get_tirx_op("exp")


class Exp2(UnaryOpWithBiasScale):
    """Compute base-2 exponential (2^x) of all elements in src and store to dst.

    If bias and scale are provided: dst = exp2(src * scale + bias)
    """

    op = get_tirx_op("exp2")


class Select(BinaryOp):
    """Select elements from src1 or src2 based on the predicate.

    select(dst, src1, src2, predicate)
    """

    op = get_tirx_op("select")
    predicate = ArgProperty(3)


### Compose Ops ###
class BinaryReduce(TilePrimitiveCall):
    """Combine a binary operation with a reduction operation.

    binary_reduce(binary_output, reduce_output, binary_input1, binary_input2, binary_op, reduce_op, reduce_axes, )
    """  # noqa: E501

    op = get_tirx_op("binary_reduce")

    binary_output = ArgProperty(0)
    reduce_output = ArgProperty(1)
    binary_input1 = ArgProperty(2)
    binary_input2 = ArgProperty(3)
    binary_op = ArgProperty(4)
    reduce_op = ArgProperty(5)
    reduce_axes = ArgProperty(6)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.binary_input1, self.binary_input2]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.binary_output, self.reduce_output]


class UnaryReduce(TilePrimitiveCall):
    """Combine a unary operation with a reduction operation.

    unary_reduce(unary_output, reduce_output, unary_input, unary_op, reduce_op, bias, scale, reduce_axes)
    """  # noqa: E501

    op = get_tirx_op("unary_reduce")

    unary_output = ArgProperty(0)
    reduce_output = ArgProperty(1)
    unary_input = ArgProperty(2)
    unary_op = ArgProperty(3)
    reduce_op = ArgProperty(4)
    bias = ArgProperty(5)
    scale = ArgProperty(6)
    reduce_axes = ArgProperty(7)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.unary_input, self.bias, self.scale]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.unary_output, self.reduce_output]


class BinaryChain(TilePrimitiveCall):
    """Chain multiple binary operations together.

    binary_chain(output, data, operand0, operand1, op0, op1, reverse1)

    if not reverse1:
        output = (operand0 op0 data) op1 operand1
    else:
        output = operand1 op1 (operand0 op0 data)
    """

    op = get_tirx_op("binary_chain")

    output = ArgProperty(0)
    data = ArgProperty(1)
    operand0 = ArgProperty(2)
    operand1 = ArgProperty(3)
    op0 = ArgProperty(4)
    op1 = ArgProperty(5)
    reverse1 = ArgProperty(6)

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.data, self.operand0, self.operand1]

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.output]


class ReduceNegate(ReduceOp):
    """
    Negate the result of a reduction operation.

    reduce_negate(output, input, reduce_axes, accum, reduce_op)
    """

    op = get_tirx_op("reduce_negate")

    reduce_op = ArgProperty(4)


class ComposeOp(TilePrimitiveCall):
    """Generic operator for composition of multiple operations.

    Must be lowered to specific compose operations before operator-level passes.
    """

    # TODO: add a pass to lower generic compose_op to specific compose ops

    op = get_tirx_op("compose_op")

    @property
    def srcs(self) -> list[Expr]:
        """Get the source expressions (inputs) of the operator."""
        raise NotImplementedError(
            "Generic compose_op must be lowered to specific compose ops before operator-level passes"  # noqa: E501
        )

    @property
    def dsts(self) -> list[Expr]:
        """Get the destination expressions (outputs) of the operator."""
        raise NotImplementedError(
            "Generic compose_op must be lowered to specific compose ops before operator-level passes"  # noqa: E501
        )


class PermuteLayout(TilePrimitiveCall):
    """Move data so the buffer's bytes are arranged under a different layout.

    Logical shape is preserved; only the byte placement changes. ``dst`` and
    ``src`` carry their own TileLayouts; on lowering, the dispatcher reads
    those layouts and emits a register-staged warp transpose, optionally
    inserting a bank-conflict-avoiding XOR-swizzle on the per-lane register
    slots.

    Args: ``permute_layout(dst_region, src_region)``.
    ``dst`` and ``src`` may alias the same underlying SMEM (in-place).
    """

    op = get_tirx_op("permute_layout")

    @property
    def dst(self) -> Expr:
        return self.args[0]

    @property
    def src(self) -> Expr:
        return self.args[1]

    @property
    def srcs(self) -> list[Expr]:
        return [self.src]

    @property
    def dsts(self) -> list[Expr]:
        return [self.dst]
