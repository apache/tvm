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
# pylint: disable=invalid-name
"""Hexagon-specific IR transformations"""

import functools as ft
import numpy as np

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    is_tuple,
    rewrite,
    wildcard,
)
from tvm.topi.utils import get_const_tuple
from tvm.relay.expr import Call
from tvm.runtime import ndarray as nd
from ..._ffi.registry import register_func

### VTCM

vtcm_size = 4 * 1024 * 1024  # pylint: disable=invalid-name


@register_func("tvm.info.mem.local.vtcm")
def mem_info_vtcm():
    # pylint: disable=bad-whitespace
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_num_bits=vtcm_size * 8,
        max_simd_bits=128 * 8,
        head_address=tvm.runtime.const(100, "uint32"),
    )


def lower_vtcm_(get_alloc, get_free, def_align, func, mod, ctx):  # pylint: disable=unused-argument
    """Generic VTCM allocation

    Parameters
    ----------
    get_alloc : function: tir.Allocate, int -> tir.expr (dtype='handle')
      The VTCM allocation function. It takes an Allocate statement, and the required
      alignment, and returns a pointer to the allocated VTCM buffer.
    get_free : function: tir.expr (dtype='handle') -> None
      The VTCM deallocation function. It takes the address of the allocated buffer
      and frees it. It returns no value.
    def_align : int
      The default alignment that will be passed to the allocation function, if the
      program does not specify the alignment via a 'storage_alignment' attribute.
    func : tir.PrimFunc
    mod : tvm.IRModule
    ctx : transform.PassContext

    Returns
    -------
    stmt : tvm.stmt
        Transformed function body.
    """

    vtcm_buffers = []
    alignments = {}

    def buf_align(var):
        """Determine the alignment of the buffer with variable 'var'."""
        if var in alignments and alignments[var]:
            return alignments[var][-1]
        return def_align

    def visit(stmt):
        """Collect information about VTCM buffers and their alignments."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "storage_alignment":
                if not stmt.node in alignments:
                    alignments[stmt.node] = []
                alignments[stmt.node].append(stmt.value)
        elif isinstance(stmt, tvm.tir.Allocate):
            scope = stmt.buffer_var.type_annotation.storage_scope
            if scope == "local.vtcm":
                vtcm_buffers.append(stmt.buffer_var)

    def mutate(stmt):
        """Insert calls to VTCM allocation and deallocation routines."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "storage_alignment":
                alignments[stmt.node].pop()
            return stmt
        if isinstance(stmt, tvm.tir.Allocate):
            var = stmt.buffer_var
            scope = var.type_annotation.storage_scope
            is_vtcm = var in vtcm_buffers
            if scope == "local.vtcm":
                vtcm_buffers.pop()
            if is_vtcm:
                is_null = tvm.tir.call_intrin("bool", tvm.ir.Op.get("tir.isnullptr"), var)
                throw_error = tvm.tir.call_intrin(
                    "int32", tvm.ir.Op.get("tir.tvm_throw_last_error")
                )
                body_w_free = tvm.tir.SeqStmt([stmt.body, tvm.tir.Evaluate(get_free(var))])
                body_w_check = tvm.tir.IfThenElse(
                    is_null, tvm.tir.Evaluate(throw_error), body_w_free
                )
                return tvm.tir.LetStmt(
                    stmt.buffer_var, get_alloc(stmt, buf_align(var)), body_w_check
                )
            return stmt
        raise ValueError("Wrong argument type (" + type(stmt) + ") to 'mutate'")

    f = func.with_body(
        tvm.tir.stmt_functor.ir_transform(
            func.body, visit, mutate, ["tir.Allocate", "tir.AttrStmt"]
        )
    )
    return f


def ir_lower_vtcm():
    """Create a VTCM lowering pass.

    VTCM memory has to be allocated using special functions.
    """

    def get_alloc(stmt, align):
        assert isinstance(stmt, tvm.tir.Allocate)
        return tvm.tir.call_extern(
            "handle",
            "HexagonBackendAllocateVTCM",
            ft.reduce(lambda x, y: x * y, stmt.extents, 1),
            align,
        )

    def get_free(var):
        return tvm.tir.call_extern("handle", "HexagonBackendFreeVTCM", var)

    # pylint: disable=bad-whitespace
    @tvm.tir.transform.prim_func_pass(opt_level=0, name="Lower VTCM pass")
    def transform(func, mod, ctx):
        return lower_vtcm_(get_alloc, get_free, 2048, func, mod, ctx)

    return transform


def ir_lower_vtcm_pass():
    return [(3, ir_lower_vtcm())]


class qdistilbert_rewrite(DFPatternCallback):
    """
    A callback to replace the below pattern:
    Pattern:
    %35 = strided_slice(%34, begin=[0, 0, 0], end=[1, 128, 64], strides=[1, 1, 1], axes=None);
    %44 = reshape(%35, newshape=[-1, 64]);
    <snip>
    %42 = strided_slice(%41, begin=[0, 0, 0], end=[1, 64, 128], strides=[1, 1, 1], axes=None);
    %43 = reshape(%42, newshape=[64, 128]);
    %45 = transpose(%43, axes=[1, 0]);
    <snip>
    %46 = qnn.dense(%44, %45, 13, 1, 0.0541715f, 0.0489368f, units=None, out_dtype="int32");
    %47 = qnn.requantize(%46, 0.00265098f, 0, 0.728874f, -14, axis=1, out_dtype="int8");
    <snip>
    %125 = expand_dims(%47, axis=0) /* ty=Tensor[(1, 128, 128), int8] */;
    < The above pattern repeats 12 times, which is the batch size >

    %137 = (%125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136);
    %138 = concatenate(%137);

    """

    def __init__(self):
        super(qdistilbert_rewrite, self).__init__()
        self.A = wildcard()  # Tensor A
        self.B = wildcard()  # Tensor B
        self.batch = 12  # Number of time pattern repeats or Batch size

        self.d = []  # List of dense quantization parameters
        self.q = []  # List of requantize parameters
        L = []  # List of patterns

        z = tvm.tir.IntImm("int64", 0)
        s1 = tvm.tir.IntImm("int64", 1)

        for i in range(self.batch):
            x = tvm.tir.IntImm("int64", i)

            self.d.append([is_constant(), is_constant(), is_constant(), is_constant()])
            self.q.append([is_constant(), is_constant(), is_constant(), is_constant()])

            pat_a = is_op("strided_slice")(self.A).has_attr(
                {"begin": [x, z, z], "strides": [s1, s1, s1]}
            )
            pat_a = is_op("reshape")(pat_a)

            pat_b = is_op("strided_slice")(self.B).has_attr(
                {"begin": [x, z, z], "strides": [s1, s1, s1]}
            )
            pat_b = is_op("reshape")(pat_b)
            pat_b = is_op("transpose")(pat_b)

            pat = is_op("qnn.dense")(
                pat_a, pat_b, self.d[i][0], self.d[i][1], self.d[i][2], self.d[i][3]
            )
            pat = is_op("qnn.requantize")(
                pat, self.q[i][0], self.q[i][1], self.q[i][2], self.q[i][3]
            )
            pat = is_op("expand_dims")(pat)
            L.append(pat)

        T = is_tuple(L)
        self.pattern = is_op("concatenate")(T)

    def check_quant_params(self, node_map):
        """checking if dense and requant params are the same across patterns"""
        r = self.batch
        x1 = [node_map[self.d[0][i]][0].data.numpy().item() for i in range(4)]
        x2 = [node_map[self.q[0][i]][0].data.numpy().item() for i in range(4)]
        for i in range(1, r):
            for j in range(4):
                y1 = node_map[self.d[i][j]][0].data.numpy().item()
                y2 = node_map[self.q[i][j]][0].data.numpy().item()
                if x1[j] != y1 or x2[j] != y2:
                    return False
        return True

    def callback(self, pre, post, node_map):
        A = node_map[self.A][0]
        B = node_map[self.B][0]

        if not self.check_quant_params(node_map):
            return post

        [a0, a1, a2] = [0, 0, 0]  # Tensor A shape
        [b0, b1, b2] = [0, 0, 0]  # Tensor B shape

        if isinstance(A, relay.expr.Call) and isinstance(B, relay.expr.Call):
            if A.checked_type is None or B.checked_type is None:
                # Need infer pass to be run before this pass
                return post
            if len(A.checked_type.shape) == 3 and len(B.checked_type.shape) == 3:
                [a0, a1, a2] = A.checked_type.shape
                [b0, b1, b2] = B.checked_type.shape

        if isinstance(A, relay.Var) and isinstance(B, relay.Var):
            if len(A.type_annotation.shape) == 3 and len(B.type_annotation.shape) == 3:
                [a0, a1, a2] = A.type_annotation.shape
                [b0, b1, b2] = B.type_annotation.shape

        # Check if the batch size is same as expected tensor size
        if (a0 != self.batch) or (b0 != self.batch):
            return post

        for i in range(self.batch):
            # end=(x, pa1, pa2) attribute of strided_slice for Tensor A
            pa1 = pre.args[0][i].args[0].args[0].args[0].args[0].attrs.end[1].value
            pa2 = pre.args[0][i].args[0].args[0].args[0].args[0].attrs.end[2].value

            # end=(x, pb1, pb2) attribute of strided_slice for Tensor B
            pb1 = pre.args[0][i].args[0].args[0].args[1].args[0].args[0].attrs.end[1].value
            pb2 = pre.args[0][i].args[0].args[0].args[1].args[0].args[0].attrs.end[2].value

            if a1 != pa1 or a2 != pa2 or b1 != pb1 or b2 != pb2:
                return post

        d = [node_map[self.d[0][i]][0] for i in range(4)]
        q = [node_map[self.q[0][i]][0] for i in range(4)]

        out = relay.op.transpose(B, axes=[0, 2, 1])
        out = relay.qnn.op.batch_matmul(A, out, d[0], d[1], d[2], d[3], out_dtype="int32")
        out = relay.qnn.op.requantize(out, q[0], q[1], q[2], q[3], out_dtype="int8")
        return out


def rewrite_qdistilbert(mod):
    """Rewrite the Quantized Distilbert to reduce computational complexity."""
    mod["main"] = rewrite(qdistilbert_rewrite(), mod["main"])
    return mod


class remove_empty_pad_callback(DFPatternCallback):
    """
    A callback to remove empty pad op from the below pattern:
    Pattern:
    %0 = cast(0f, dtype="float16");
    %1 = nn.pad(%inp, %0, pad_width=[[0i64, 0i64], [0i64, 0i64]]);
    nn.matmul(%1, %inp2, units=None)

    """

    def __init__(self):
        super(remove_empty_pad_callback, self).__init__()
        self.A = wildcard()
        self.B = wildcard()
        self.a = is_op("nn.pad")(self.A, wildcard()).has_attr({"pad_width": ((0, 0), (0, 0))})
        self.pattern = is_op("nn.matmul")(self.a, self.B)

    def callback(self, pre, post, node_map):
        A = node_map[self.A][0]
        B = node_map[self.B][0]
        return relay.nn.matmul(A, B)


def remove_empty_pad(mod):
    """Remove the empty pad operator."""
    mod["main"] = rewrite(remove_empty_pad_callback(), mod["main"])
    return mod


class simplify_qnn_concat_in_func(DFPatternCallback):

    """
    Propagate qnn.concat's quantization params to its inputs,
    and try to avoid redundant requantization while doing so.

    Replace
    def @main(%q1: Tensor[(1, 64, 35, 35), uint8],
        %q2: Tensor[(1, 64, 35, 35), uint8], %q3: Tensor[(1, 32, 35, 35), uint8]) {
        %0 = nn.max_pool2d(%q1, pool_size=[3, 3], padding=[1, 1, 1, 1], layout="NHWC");
        %1 = qnn.requantize(%q2, 0.000109401f, 0, 0.00345f, 0, axis=1, out_dtype="uint8");
        %2 = (%0, %1, %q3);
        %3 = (0.0425042f, 0.00345f, 0.0486874f);
        %4 = (0, 0, 0);
        qnn.concatenate(%2, %3, %4, 0.0486874f, 0, axis=1)
    }

    with

    def @main(%q1: Tensor[(1, 64, 35, 35), uint8],
        %q2: Tensor[(1, 64, 35, 35), uint8], %q3: Tensor[(1, 32, 35, 35), uint8]) {
        %0 = nn.max_pool2d(%q1, pool_size=[3, 3], padding=[1, 1, 1, 1], layout="NHWC");
        %1 = qnn.requantize(%0, 0.0425042f, 0, 0.0486874f, 0, axis=1, out_dtype="uint8");
        %2 = qnn.requantize(%q2, 0.000109401f, 0, 0.0486874f, 0, axis=1, out_dtype="uint8");
        %3 = (%1, %2, %q3);
        concatenate(%3, axis=1)
    }
    """

    def __init__(self):
        super(simplify_qnn_concat_in_func, self).__init__()
        self.qvals = wildcard()
        self.scales = wildcard()
        self.zps = wildcard()
        self.out_scale = wildcard()
        self.out_zp = wildcard()
        self.pattern = is_op("qnn.concatenate")(
            self.qvals, self.scales, self.zps, self.out_scale, self.out_zp
        )

    def callback(self, pre, post, node_map):
        in_qvals = node_map[self.qvals][0]
        in_scales = node_map[self.scales][0]
        in_zps = node_map[self.zps][0]
        new_qvals = []
        for i in range(len(in_qvals)):
            new_requant_args = []
            # TODO Generalize for all qnn ops
            if isinstance(in_qvals[i], Call) and (in_qvals[i].op.name == "qnn.requantize"):
                # propagate scale/zp of qnn.concat to this requantize op
                for j in range(3):
                    new_requant_args.append(in_qvals[i].args[j])
                new_requant_args += [node_map[self.out_scale][0], node_map[self.out_zp][0]]
                new_qvals.append(relay.qnn.op.requantize(*new_requant_args, **(in_qvals[i].attrs)))
            else:
                # simply create a new requantize op if there is a change in quantization params
                # if not, just retain the old qval
                if (in_scales[i] == node_map[self.out_scale][0]) and (
                    in_zps[i] == node_map[self.out_zp][0]
                ):
                    new_qvals.append(in_qvals[i])
                else:
                    new_requant_args += [
                        in_qvals[i],
                        in_scales[i],
                        in_zps[i],
                        node_map[self.out_scale][0],
                        node_map[self.out_zp][0],
                    ]
                    new_qvals.append(
                        relay.qnn.op.requantize(
                            *new_requant_args,
                            axis=post.attrs["axis"],
                            out_dtype=post.checked_type.dtype,
                        )
                    )

        new_op = relay.op.concatenate(
            new_qvals,
            node_map[self.pattern][0].attrs["axis"],
        )
        return new_op


# Right now context is ignored
@tvm.transform.module_pass(opt_level=1)
def simplify_qnn_concat(mod, _=None):
    for global_var in mod.functions.keys():
        mod[global_var] = rewrite(simplify_qnn_concat_in_func(), mod[global_var])
    return mod


class simplify_conv_pat_in_func(DFPatternCallback):

    """
    Simplify Mul->Sub->Conv->bias_add to Conv->bias_add->add sequence if
    one of the inputs to Mul and Sub are constant scalars.

    Replace
    def @main(%q1: Tensor[(1, 128, 128, 3), float16])
        %0 = multiply(%q1, c1_const_scalar)  /* ty=Tensor[(1, 128, 128, 3), float16] */;
        %1 = subtract(%0, c2_const_scalar) /* ty=Tensor[(1, 128, 128, 3), float16] */
        %2 = transpose(%1, axes=[0,3,1,2])
            /* ty=Tensor[(1, 3, 128, 128), float16] */
        %3 = nn.conv2d(%2, weights, ...) .
        %4 = nn.bias_add(%3, bias)
    }

    with

    def @main(%q1: Tensor[(1, 128, 128, 3), float16])
        %0 = transpose(%q1, axes=[0, 3, 1, 2])
            /* ty=Tensor[(1, 3, 128, 128), float16] */;
        %1 = multiply(c1, weights) /* ty=Tensor[(64, 3, 3, 3), float16] */;
        %2 = nn.conv2d(%0, %1, padding=[1, 1, 1, 1],
            channels=64, kernel_size=[3, 3])
            /* ty=Tensor[(1, 64, 128, 128), float16] */;
        %3 = subtract(%0 shaped zero_tensor, c2)
            /* ty=Tensor[(1, 3, 128, 128), float16] */;
        %4 = nn.bias_add(%2, bias) /* ty=Tensor[(1, 64, 128, 128), float16] */;
        %5 = nn.conv2d(%3, weights, padding=[1, 1, 1, 1],
            channels=64, kernel_size=[3, 3])
            /* ty=Tensor[(1, 64, 128, 128), float16] */;
        add(%4, %5) /* ty=Tensor[(1, 64, 128, 128), float16] */

    Why is it legal? Ignore the transpose in the above pattern.
    res[p,q,r,s] = Conv(a*c1 - c2, W)
                 = SUM{i=[0,c-1], j=[0,kh-1], k=[0,kw-1]}
                    {(a[p,i,r+j,s+k] * c1 - c2) * W[q,i,j,k]}
                 = SUM{i=[0,c-1], j=[0,kh-1], k=[0,kw-1]}
                    {a[p,i,r+j,s+k] * c1 * W[q,i,j,k]} - c2 * W[q,i,j,k]}
                 = Conv(a, W*c1) + Conv(0-c2, W)


    }

    In the above, %1, %3, %5 are constants and can be folded, so we're
    left with 4 ops, as opposed to the original 5 ops
    """

    def __init__(self):
        super().__init__()
        self.inp = wildcard()
        self.mul = is_op("multiply")(self.inp, is_constant().has_shape(()))
        self.sub = is_op("subtract")(self.mul, is_constant().has_shape(()))
        self.act = is_op("transpose")(self.sub)
        self.weights = is_constant()
        self.conv2d_op = is_op("nn.conv2d")(self.act, self.weights)
        self.pattern = is_op("nn.bias_add")(self.conv2d_op, is_constant())

    def callback(self, pre, post, node_map):
        new_transpose = relay.transpose((node_map[self.inp][0]), **((node_map[self.act][0]).attrs))
        new_weights = relay.multiply((node_map[self.mul][0].args[1]), (node_map[self.weights][0]))
        new_conv2d = relay.nn.conv2d(
            new_transpose, new_weights, **((node_map[self.conv2d_op][0]).attrs)
        )
        new_bias_add = relay.nn.bias_add(new_conv2d, (node_map[self.pattern][0].args[1]))

        zero_tensor = relay.Constant(
            nd.array(
                np.zeros(
                    get_const_tuple((node_map[self.act][0]).checked_type.shape),
                    dtype=(node_map[self.act][0]).checked_type.dtype,
                )
            )
        )
        negated = relay.subtract(zero_tensor, (node_map[self.sub][0].args[1]))
        const_conv2d = relay.nn.conv2d(
            negated, (node_map[self.weights][0]), **((node_map[self.conv2d_op][0]).attrs)
        )
        return relay.add(new_bias_add, const_conv2d)


# Right now context is ignored
@tvm.transform.module_pass(opt_level=1)
def simplify_conv_pat(mod, _=None):
    """top level function for conv pattern simplification"""
    for global_var in mod.functions.keys():
        mod[global_var] = rewrite(simplify_conv_pat_in_func(), mod[global_var])
    return mod
