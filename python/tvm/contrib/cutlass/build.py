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
# pylint: disable=invalid-name, dangerous-default-value, arguments-differ
"""Driver for partitioning and building a Relay module for CUTLASS offload."""
import itertools
import logging
import multiprocessing
import operator
import os
from functools import reduce
from typing import Optional, Sequence

import tvm
from tvm import relax, relay, runtime
from tvm._ffi.registry import register_func
from tvm.contrib.nvcc import get_cuda_version
from tvm.topi.utils import get_const_tuple

from .gen_conv2d import CutlassConv2DProfiler
from .gen_gemm import CutlassGemmProfiler
from .library import ConvKind, LayoutType

logger = logging.getLogger("cutlass")


def has_cutlass():
    """Returns true if the CUTLASS custom codegen is available"""
    return tvm.get_global_func("relay.ext.cutlass.create_c_source_module", True) is not None


def _get_cutlass_path():
    invalid_paths = []
    for rel in ["../../../../", "../../../", "../../"]:
        tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel)
        cutlass_path = os.path.join(tvm_root, "3rdparty/cutlass")
        if os.path.exists(cutlass_path):
            return cutlass_path
        invalid_paths.append(cutlass_path)
    raise AssertionError(f"The CUTLASS root directory not found in: {invalid_paths}")


def _get_cutlass_compile_options(sm, threads, use_fast_math=False):
    cutlass_root = _get_cutlass_path()
    cutlass_include = os.path.join(cutlass_root, "include")
    cutlass_util_include = os.path.join(cutlass_root, "tools/util/include")
    cutlass_attention_include = os.path.join(cutlass_root, "examples/41_fused_multi_head_attention")
    cutlass_fpA_intB_gemm_include = os.path.join(cutlass_root, "../cutlass_fpA_intB_gemm")

    kwargs = {}
    kwargs["cc"] = "nvcc"
    kwargs["options"] = [
        "-c",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        f"-gencode=arch=compute_{sm},code=[sm_{sm},compute_{sm}]",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-Xcompiler=-fvisibility=hidden",
        "-O3",
        "-std=c++17",
        f"-I{cutlass_include}",
        f"-I{cutlass_util_include}",
        f"-I{cutlass_attention_include}",
        f"-I{cutlass_fpA_intB_gemm_include}",
    ]
    if use_fast_math:
        kwargs["options"].append("-DCUTLASS_USE_TANH_FOR_SIGMOID")
    cuda_ver = get_cuda_version()
    if cuda_ver >= (11, 2):
        ncpu = multiprocessing.cpu_count() if threads < 0 else threads
        kwargs["options"].append(f"-t {ncpu}")
    return kwargs


class OpAnnotator(tvm.relay.ExprVisitor):
    """Annotates partitioned functions with shape and dtype information."""

    def __init__(self):
        super().__init__()
        self.signature = {}

    def visit_call(self, call):
        op = call.op
        if isinstance(op, relay.Function) and "Composite" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            for i, arg in enumerate(op.params):
                self.signature[f"arg{i}_shape"] = arg.checked_type.shape
                self.signature[f"arg{i}_dtype"] = arg.checked_type.dtype
            self.signature["ret_shape"] = op.ret_type.shape
            self.signature["ret_dtype"] = op.ret_type.dtype
            self.visit(op.body)

        elif isinstance(op, tvm.ir.Op) and op.name in [
            "nn.conv2d",
            "nn.conv2d_transpose",
            "nn.conv2d_backward_weight",
        ]:
            self.op_attrs = call.attrs

        for arg in call.args:
            self.visit(arg)


def select_gemm_kernel(
    cutlass_profiler,
    op_type,
    MM,
    KK,
    NN,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    use_3xtf32,
    batched,
    find_first_valid,
    use_multiprocessing,
):
    """Run CUTLASS profiler to select the best kernel, or return the default one for dynamic
    workloads."""
    if any(isinstance(s, tvm.tir.Any) for s in [MM, KK, NN]):
        out = cutlass_profiler.get_default(
            op_type, out_dtype, arg0_dtype, arg1_dtype, use_3xtf32, batched=batched
        )
        name, cutlass_op_def = out["name"], out["opdef"]
        logger.info("Picked the default kernel %s", name)
    else:
        name, cutlass_op_def, _ = cutlass_profiler.profile(
            op_type,
            MM,
            NN,
            KK,
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            use_3xtf32,
            batched=batched,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
        )
        if not find_first_valid:
            logger.info("The best kernel is %s", name)
        else:
            logger.info("Picked the first kernel found %s", name)

    return name, cutlass_op_def


def handle_batch_matmul(
    cutlass_profiler,
    op_type,
    arg0_shape,
    arg1_shape,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    use_3xtf32,
    find_first_valid,
    use_multiprocessing,
):
    """Profile and select a kernel for batch_matmul op workload."""
    MM = arg0_shape[1]
    KK = arg0_shape[2]
    NN = arg1_shape[1]

    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        True,
        find_first_valid,
        use_multiprocessing,
    )

    return {
        "batch": arg0_shape[0],
        "batch_stride_A": arg0_shape[1] * arg0_shape[2],
        "batch_stride_B": arg1_shape[1] * arg1_shape[2],
        "batch_stride_C": arg0_shape[1] * arg1_shape[1],
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
    }


def handle_dense(
    cutlass_profiler,
    op_type,
    arg0_shape,
    arg1_shape,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    use_3xtf32,
    find_first_valid,
    use_multiprocessing,
):
    """Profile and select a kernel for dense op workload."""
    MM = arg0_shape[0]
    KK = arg0_shape[1]
    NN = arg1_shape[0]

    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        False,
        find_first_valid,
        use_multiprocessing,
    )

    assert "tn_align" in name, "Only supports (row_major, col_major) input layout for now."

    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
    }


def handle_conv2d(
    cutlass_profiler,
    op_type,
    d_shape,
    w_shape,
    padding,
    strides,
    dilation,
    out_dtype,
    data_dtype,
    weight_dtype,
    use_3xtf32,
    split_k_slices,
    profile_all_alignments,
    find_first_valid,
    use_multiprocessing,
):
    """Profile and select a kernel for conv2d op workload."""
    if "conv2d_transpose" in op_type:
        conv_kind = ConvKind.Dgrad
    elif "backward_weight" in op_type:
        conv_kind = ConvKind.Wgrad
    else:
        conv_kind = ConvKind.Fprop

    if any(isinstance(s, tvm.tir.Any) for s in d_shape):
        out = cutlass_profiler.get_default(
            op_type, out_dtype, data_dtype, weight_dtype, use_3xtf32, conv_kind, strides
        )
        name, cutlass_op_def = out["name"], out["opdef"]
        logger.info("Picked the default kernel %s", name)
    else:
        name, cutlass_op_def, _ = cutlass_profiler.profile(
            op_type,
            d_shape,
            w_shape,
            padding,
            strides,
            dilation,
            out_dtype,
            data_dtype,
            weight_dtype,
            use_3xtf32,
            conv_kind,
            split_k_slices,
            profile_all_alignments,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
        )
        if not find_first_valid:
            logger.info("The best kernel is %s", name)
        else:
            logger.info("Picked the first kernel found %s", name)

    return {"cutlass_op_def": cutlass_op_def, "cutlass_op_name": name}


def num_cutlass_partitions(mod):
    return sum([(1 if "cutlass" in var.name_hint else 0) for var in mod.get_global_vars()])


def tune_cutlass_kernels(
    mod,
    sm,
    use_3xtf32=True,
    split_k_slices=[1],
    profile_all_alignments=False,
    find_first_valid=False,
    use_multiprocessing=False,
    tmp_dir="./tmp",
):
    """Given a module partitioned for CUTLASS offloading, profile each workload to select which
    kernels to emit.

    Parameters
    ----------
    mod : IRModule
        The Relay module with cutlass partitions.

    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    use_3xtf32 : bool
        Wheter or not use slower but very accurate (compared to tf32) 3xtf32 mode for
        fp32 inputs on tensorcore.

    split_k_slices : list of int
        Split factor candidates for split-K GEMM. If split-K > 1, the GEMM K-loop is computed in
        parallel across split-K blocks, and a separate global reduction kernel is launched to
        accumulate partial reductions. The profiler will pick the best split-k factor from the
        given candidate list. Note that the larger split-K factor requires a larger workspace.
        Currently, parallel split-k has been tested only for wgrad. For GEMM and other conv2d
        kinds, split_k_slices is ignored.

    profile_all_alignments : bool
        When True, profile all kernal variants with smaller alignments than the largest possible.

    find_first_valid : bool
        Whether or not profile all candidate kernels, or stop profiling after
        the first applicable kernel is found.

    use_multiprocessing : bool
        Whether or not compile profiler executables for different kernels in parallel.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    mod : IRModule
        The updated module annotated with cutlass profiling information.

    num_cutlass_partition : int
        The number of partitioned functions created for CUTLASS.
    """
    gemm_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), tmp_dir)
    conv2d_profiler = CutlassConv2DProfiler(sm, _get_cutlass_path(), tmp_dir)
    num_cutlass_partition = 0
    for var in mod.get_global_vars():
        fun_name = var.name_hint
        func = mod[fun_name]
        if "cutlass" in fun_name:
            num_cutlass_partition += 1
            new_func = tune_cutlass_function(
                func,
                use_3xtf32,
                split_k_slices,
                profile_all_alignments,
                find_first_valid,
                use_multiprocessing,
                gemm_profiler,
                conv2d_profiler,
            )
            mod.update_func(var, new_func)

    return mod, num_cutlass_partition


def tune_cutlass_function(
    func,
    use_3xtf32,
    split_k_slices,
    profile_all_alignments,
    find_first_valid,
    use_multiprocessing,
    gemm_profiler,
    conv2d_profiler,
):
    """Given a function intended to be offloaded to CUTLASS,  profile each workload to select which
    kernels to emit.

    Parameters
    ----------
    func : IRModule
        The Relay Function to tune for.

    use_3xtf32 : bool
        Wheter or not use slower but very accurate (compared to tf32) 3xtf32 mode for
        fp32 inputs on tensorcore.

    split_k_slices : list of int
        Split factor candidates for split-K GEMM. If split-K > 1, the GEMM K-loop is computed in
        parallel accross split-K blocks, and a seperate global reduction kernel is launched to
        accumulate partial reductions. The profiler will pick the best split-k factor from the
        given candidate list. Note that the larger split-K factor requires a larger workspace.
        Currently, parallel split-k has been tested only for wgrad. For GEMM and other conv2d
        kinds, split_k_slices is ignored.

    profile_all_alignments : bool
        When True, profile all kernal variants with smaller alignments than the largest possible.

    find_first_valid : bool
        Whether or not profile all candidate kernels, or stop profiling after
        the first applicable kernel is found.

    use_multiprocessing : bool
        Whether or not compile profiler executables for different kernels in parallel.

    gemm_profiler : CutlassGemmProfiler
        Profiler for dense operators. May cache results between tuned functions.

    conv2d_profiler : CutlassConv2DProfiler
        Profiler for conv2d operators. May cach results between tuned functions.

    Returns
    -------
    annot_func : Function
        The input function with attributes capturing the best CUTLASS kernel found by tuning.
    """
    annotator = OpAnnotator()
    annotator.visit(func)
    out_shape = annotator.signature["ret_shape"]
    out_dtype = annotator.signature["ret_dtype"]
    op_type = annotator.signature["op_type"]

    new_attrs = {"op_type": op_type}
    new_attrs.update(annotator.signature)
    new_attrs.update(func.attrs)
    arg0_shape = new_attrs["arg0_shape"]
    arg1_shape = new_attrs["arg1_shape"]
    arg0_dtype = new_attrs["arg0_dtype"]
    arg1_dtype = new_attrs["arg1_dtype"]

    if "conv2d" in op_type:
        new_attrs["padding"] = annotator.op_attrs.padding
        new_attrs["strides"] = annotator.op_attrs.strides
        new_attrs["dilation"] = annotator.op_attrs.dilation

        if "conv2d_transpose" in op_type:
            d_shape = out_shape
            w_shape = arg1_shape
        elif "conv2d_backward_weight" in op_type:
            d_shape = arg1_shape
            w_shape = out_shape
        else:
            d_shape = arg0_shape
            w_shape = arg1_shape

        new_attrs.update(
            handle_conv2d(
                conv2d_profiler,
                op_type,
                d_shape,
                w_shape,
                annotator.op_attrs.padding,
                annotator.op_attrs.strides,
                annotator.op_attrs.dilation,
                out_dtype,
                arg0_dtype,
                arg1_dtype,
                use_3xtf32,
                split_k_slices,
                profile_all_alignments,
                find_first_valid,
                use_multiprocessing,
            )
        )
    elif "batch_matmul" in op_type:
        new_attrs.update(
            handle_batch_matmul(
                gemm_profiler,
                op_type,
                arg0_shape,
                arg1_shape,
                out_dtype,
                arg0_dtype,
                arg1_dtype,
                use_3xtf32,
                find_first_valid,
                use_multiprocessing,
            )
        )
    elif "dense" in op_type:
        new_attrs.update(
            handle_dense(
                gemm_profiler,
                op_type,
                arg0_shape,
                arg1_shape,
                out_dtype,
                arg0_dtype,
                arg1_dtype,
                use_3xtf32,
                find_first_valid,
                use_multiprocessing,
            )
        )
    else:
        raise ValueError(f"{op_type} unsupported composite")

    new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
    return relay.Function(
        func.params,
        func.body,
        ret_type=func.ret_type,
        type_params=func.type_params,
        attrs=new_attrs,
    )


def _get_call_node(expr: relax.Expr, op_name: str) -> Optional[relax.Call]:
    node = None

    def fvisit(e):
        nonlocal node
        if isinstance(e, relax.Call) and e.op.name == op_name:
            node = e

    relax.analysis.post_order_visit(expr, fvisit)
    return node


def _extract_relax_function_signature(f):
    signature = {}

    for i, arg in enumerate(f.params):
        sinfo = arg.struct_info
        if isinstance(sinfo, relax.TensorStructInfo):
            signature["arg%d_shape" % i] = get_const_tuple(sinfo.shape)
            signature["arg%d_dtype" % i] = sinfo.dtype
        elif isinstance(sinfo, relax.ShapeStructInfo):
            signature["arg%d_shape" % i] = get_const_tuple(sinfo.values)
        else:
            raise NotImplementedError()

    ret_sinfo = f.ret_struct_info
    if ret_sinfo.shape is not None:
        signature["ret_shape"] = get_const_tuple(ret_sinfo.shape)
    else:
        signature["ret_shape"] = None
    signature["ret_dtype"] = ret_sinfo.dtype

    return signature


def _extract_arg_idx(pattern_name, f):
    extract_func = tvm.get_global_func("relax.contrib.extract_arg_idx")
    arg_indices = extract_func(pattern_name, f)
    return {k: int(v) for k, v in arg_indices.items()}


def is_shape_valid_for_cutlass_matmul(
    lhs_shape: Sequence[tvm.ir.PrimExpr],
    rhs_shape: Sequence[tvm.ir.PrimExpr],
) -> bool:
    """
    Check whether the shape of inputs can be handled by CUTLASS GEMM.

    The stride-based batch matmul in CUTLASS cannot handle cases that some of
    the batch dimensions need to be stretched while others don't. This means
    it can only handle ND x ND whose batch dimensions match exactly on both side,
    as well as ND x 2D and 2D x ND. For example, it cannot handle matmul with shape
    (2, 1, 4, 8) x (2, 3, 8, 16), because the batch stride of lhs is not constant.
    """
    if not isinstance(lhs_shape[-1], (tvm.tir.expr.IntImm, int)):
        # Reduction axis must be constant
        return False

    lhs_batches = reduce(operator.mul, lhs_shape[:-2], 1)
    rhs_batches = reduce(operator.mul, rhs_shape[:-2], 1)
    if lhs_batches == 1 or rhs_batches == 1:
        # This could be regular matmul or batch matmul with shape ND x 2D or 2D x ND
        return True

    analyzer = tvm.arith.Analyzer()
    # If one side has less dimensions, use 1 to fill the gap
    batch_dim_pairs = list(
        itertools.zip_longest(
            list(lhs_shape)[-3::-1],  # Remove the last two dimensions and reverse
            list(rhs_shape)[-3::-1],
            fillvalue=1,
        )
    )
    return all(analyzer.can_prove_equal(p[0], p[1]) for p in batch_dim_pairs)


@relax.expr_functor.mutator
class CutlassRelaxFunctionAnnotator(relax.PyExprMutator):
    """A Relax function mutator that tunes and annotates CUTLASS composite functions
    with shape, dtype and generated templates.
    """

    def __init__(
        self,
        mod,
        conv2d_profiler: CutlassConv2DProfiler,
        gemm_profiler: CutlassGemmProfiler,
        options,
    ):
        super().__init__(mod)
        self.options = options
        self.conv2d_profiler = conv2d_profiler
        self.gemm_profiler = gemm_profiler

    def handle_conv2d(self, f, op_type):
        """Tune and annotate a conv2d op."""
        signature = _extract_relax_function_signature(f)
        arg_idx = _extract_arg_idx(op_type, f)
        op_attrs = _get_call_node(f.body, "relax.nn.conv2d").attrs

        data_arg = f"arg{arg_idx['lhs']}"
        weight_arg = f"arg{arg_idx['rhs']}"

        d_shape = signature[f"{data_arg}_shape"]
        w_shape = signature[f"{weight_arg}_shape"]
        out_shape = signature["ret_shape"]
        data_dtype = signature[f"{data_arg}_dtype"]
        weight_dtype = signature[f"{weight_arg}_dtype"]
        out_dtype = signature["ret_dtype"]
        padding = op_attrs["padding"]
        strides = op_attrs["strides"]
        dilation = op_attrs["dilation"]
        conv_kind = ConvKind.Fprop

        use_3xtf32 = self.options.get("use_3xtf32", False)
        profile_all_alignments = self.options.get("profile_all_alignments", False)
        find_first_valid = self.options.get("find_first_valid", True)
        use_multiprocessing = self.options.get("use_multiprocessing", True)
        split_k_slices = self.options.get("split_k_slices", [1])

        op_name, op_def, _ = self.conv2d_profiler.profile(
            op_type,
            d_shape,
            w_shape,
            padding,
            strides,
            dilation,
            out_dtype,
            data_dtype,
            weight_dtype,
            use_3xtf32,
            conv_kind,
            split_k_slices,
            profile_all_alignments,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
        )

        attrs = {
            "op_type": op_type,
            "data_arg_idx": arg_idx["lhs"],
            "weight_arg_idx": arg_idx["rhs"],
            "bias_arg_idx": arg_idx.get("bias"),
            "residual_arg_idx": arg_idx.get("residual"),
            "arg0_dtype": data_dtype,
            "arg1_dtype": weight_dtype,
            "ret_dtype": out_dtype,
            "arg0_shape": d_shape,
            "arg1_shape": w_shape,
            "ret_shape": out_shape,
            "strides": strides,
            "padding": padding,
            "dilation": dilation,
            "cutlass_op_name": op_name,
            "cutlass_op_def": op_def,
        }

        residual_arg = arg_idx.get("residual")

        if residual_arg:
            residual_shape = signature[f"arg{residual_arg}_shape"]
            attrs["residual_shape"] = residual_shape
        elif "residual" in op_type:
            attrs["residual_shape"] = d_shape

        return f.with_attrs(attrs)

    def handle_decode_matmul(self, f, op_type):
        """Annotate a decode -> matmul op."""
        arg_idx = _extract_arg_idx(op_type, f)
        signature = _extract_relax_function_signature(f)
        lhs_arg = f"arg{arg_idx['lhs']}"
        rhs_arg = f"arg{arg_idx['w_encoded']}"
        lhs_shape = signature[f"{lhs_arg}_shape"]
        rhs_shape = signature[f"{rhs_arg}_shape"]
        ret_shape = signature["ret_shape"]
        N = ret_shape[-1]

        attrs = {
            "op_type": op_type,
            "lhs_arg_idx": arg_idx["lhs"],
            "rhs_arg_idx": arg_idx["w_encoded"],
            "scales_arg_idx": arg_idx["scales"],
            "bias_arg_idx": arg_idx.get("bias"),
            "batch_offset": len(lhs_shape) - 2,
            "activation": "identity",
        }

        attrs["bias_stride"] = 0

        if "bias" in arg_idx:
            bias_shape = signature[f"arg{arg_idx['bias']}_shape"]
            bias_shape_1d = reduce(operator.mul, bias_shape, 1)
            if bias_shape_1d != bias_shape[-1]:
                attrs["bias_stride"] = bias_shape[-1]

        if N == rhs_shape[1]:
            attrs["weight_nbit"] = 8
        else:
            assert N == rhs_shape[1] * 2
            attrs["weight_nbit"] = 4

        if "residual" in op_type:
            residual_pos = op_type.find("residual_")
            postfix = op_type[residual_pos + len("residual_") :]

            if postfix.startswith("multiply"):
                binary_op = "multiply"
            else:
                binary_op = "plus"

            if "relu" in postfix:
                unary_op = "relu"
            else:
                unary_op = "identity"

            activation = "identity"

            for act in ["relu", "silu", "gelu"]:
                if act in op_type[op_type.find("matmul_") + len("matmul_") : residual_pos]:
                    activation = act
                    break

            attrs.update(
                {
                    "unary_op": unary_op,
                    "binary_op": binary_op,
                    "activation": activation,
                    "residual_arg_idx": arg_idx["residual"],
                }
            )
        else:
            for act in ["relu", "silu", "gelu"]:
                if act in op_type:
                    attrs["activation"] = act
                    break

        return f.with_attrs(attrs)

    def handle_matmul(self, f, op_type):
        """Tune and annotate a matmul op."""
        signature = _extract_relax_function_signature(f)
        arg_idx = _extract_arg_idx(op_type, f)

        lhs_arg = f"arg{arg_idx['lhs']}"
        rhs_arg = f"arg{arg_idx['rhs']}"

        lhs_shape = signature[f"{lhs_arg}_shape"]
        rhs_shape = signature[f"{rhs_arg}_shape"]
        out_shape = signature["ret_shape"]
        lhs_dtype = signature[f"{lhs_arg}_dtype"]
        rhs_dtype = signature[f"{rhs_arg}_dtype"]
        out_dtype = signature["ret_dtype"]

        if not is_shape_valid_for_cutlass_matmul(lhs_shape, rhs_shape):
            raise ValueError(f"Cannot handle the input shapes, lhs: {lhs_shape}, rhs: {rhs_shape}")

        MM = lhs_shape[-2]
        KK = lhs_shape[-1]
        if "transposed" in op_type:
            NN = rhs_shape[-2]
            ldb = "K"
            layout_b = LayoutType.ColumnMajor
        else:
            NN = rhs_shape[-1]
            ldb = "N"
            layout_b = LayoutType.RowMajor

        lhs_batches = reduce(operator.mul, lhs_shape[:-2], 1)
        rhs_batches = reduce(operator.mul, rhs_shape[:-2], 1)
        if lhs_batches == 1 and rhs_batches == 1:
            # Regular matmul
            is_batched = False
            batch_attrs = {}
        else:
            is_batched = True
            batch_attrs = {
                # If both lhs_batches and rhs_batches are greater than 1,
                # they must be equal. This is checked by is_shape_valid_for_cutlass_matmul.
                "batch": lhs_batches if rhs_batches == 1 else rhs_batches,
                "batch_stride_A": 0 if lhs_batches == 1 else MM * KK,
                "batch_stride_B": 0 if rhs_batches == 1 else KK * NN,
                "batch_stride_C": MM * NN,
            }

        use_3xtf32 = self.options.get("use_3xtf32", False)
        find_first_valid = self.options.get("find_first_valid", True)
        use_multiprocessing = self.options.get("use_multiprocessing", True)

        op_name, op_def, _ = self.gemm_profiler.profile(
            op_type,
            MM,
            NN,
            KK,
            out_dtype,
            lhs_dtype,
            rhs_dtype,
            use_3xtf32,
            batched=is_batched,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
            layout_b=layout_b,
        )

        return f.with_attrs(
            {
                "op_type": op_type,
                "lhs_arg_idx": arg_idx["lhs"],
                "rhs_arg_idx": arg_idx["rhs"],
                "residual_arg_idx": arg_idx.get("residual"),
                "bias_arg_idx": arg_idx.get("bias"),
                "arg0_dtype": signature["arg0_dtype"],
                "arg1_dtype": signature["arg1_dtype"],
                "ret_dtype": out_dtype,
                "arg0_shape": signature["arg0_shape"],
                "arg1_shape": signature["arg1_shape"],
                "ret_shape": out_shape,
                "lda": "K",
                "ldb": ldb,
                "ldc": "N",
                "cutlass_op_name": op_name,
                "cutlass_op_def": op_def,
                **batch_attrs,
            }
        )

    def handle_attention(self, f, op_type):
        """Annotate an attention op."""
        signature = _extract_relax_function_signature(f)
        if _get_call_node(f.body, "relax.nn.attention") is not None:
            op_attrs = _get_call_node(f.body, "relax.nn.attention").attrs
        elif _get_call_node(f.body, "relax.nn.attention_bias") is not None:
            op_attrs = _get_call_node(f.body, "relax.nn.attention_bias").attrs
        else:
            raise ValueError("Cannot find call node for attention")
        arg = {}

        if "stacked_attention" in op_type:
            arg["arg0_shape"] = signature["arg0_shape"]
            arg["arg0_dtype"] = signature["arg0_dtype"]
            arg["arg1_shape"] = q_shape = signature["arg1_shape"]

            if "arg3_shape" not in signature:
                # arg0: qkv, arg1: shape, arg2: workspace
                arg["arg2_shape"] = k_shape = signature["arg1_shape"]
                arg["arg3_shape"] = v_shape = signature["arg1_shape"]
            else:
                # arg0: qkv, arg1: shape1, arg2: shape2, arg3: shape3, arg4: workspace
                arg["arg2_shape"] = k_shape = signature["arg2_shape"]
                arg["arg3_shape"] = v_shape = signature["arg3_shape"]

            if "arg5_dtype" in signature:
                # arg0: qkv, arg1: shape1, arg2: shape2, arg3: shape3, arg4: bias, arg5: workspace
                arg["bias_dtype"] = signature["arg4_dtype"]
            if "arg5_shape" in signature:
                arg["bias_shape"] = signature["arg4_shape"]

            qkv_layout = "qkv_stacked"
        else:
            # arg0: q, arg1: k, arg2: v,  arg3: bias, arg4: workspace
            arg["arg0_shape"] = q_shape = signature["arg0_shape"]
            arg["arg1_shape"] = k_shape = signature["arg1_shape"]
            arg["arg2_shape"] = v_shape = signature["arg2_shape"]
            arg["arg0_dtype"] = signature["arg0_dtype"]
            arg["arg1_dtype"] = signature["arg1_dtype"]
            arg["arg2_dtype"] = signature["arg2_dtype"]

            if "arg4_dtype" in signature:
                arg["bias_dtype"] = signature["arg3_dtype"]
            if "arg4_shape" in signature:
                arg["bias_shape"] = signature["arg3_shape"]

            qkv_layout = "default"

        out_shape = signature["ret_shape"]
        out_dtype = signature["ret_dtype"]
        num_batches, num_queries, num_heads, head_dim = q_shape
        _, num_keys, _, _ = k_shape
        _, _, _, head_dim_value = v_shape
        scale = op_attrs.scale

        if op_attrs.causal_mask is None:
            custom_mask_type = 0
        elif op_attrs.causal_mask == "TopLeft":
            custom_mask_type = 1
        elif op_attrs.causal_mask == "BottomRight":
            custom_mask_type = 2
        else:
            raise NotImplementedError()

        return f.with_attrs(
            {
                "op_type": op_type,
                "ret_dtype": out_dtype,
                "ret_shape": out_shape,
                "num_batches": num_batches,
                "num_queries": num_queries,
                "num_keys": num_keys,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "head_dim_value": head_dim_value,
                "scale": scale,
                "arch": self.options["sm"],
                "qkv_layout": qkv_layout,
                "custom_mask_type": custom_mask_type,
                **arg,
            }
        )

    def handle_norm(self, f, _):
        """Annotate a layer or rms norm op."""
        signature = _extract_relax_function_signature(f)
        attrs = {}
        attrs["batch_rank"] = len(signature["arg0_shape"][:-1])
        attrs["M"] = reduce(operator.mul, signature["arg0_shape"][:-1], 1)
        attrs["N"] = signature["arg0_shape"][-1]
        dtype = signature["arg0_dtype"]
        attrs["data_type"] = {"float32": "float", "float16": "cutlass::half_t"}[str(dtype)]
        return f.with_attrs(attrs)

    def visit_function_(self, f):
        if "Composite" not in f.attrs:
            body = super().visit_expr(f.body)
            return relax.Function(f.params, body, f.ret_struct_info, f.is_pure, f.attrs, f.span)

        op_type = f.attrs["Composite"]

        if "conv2d" in op_type:
            return self.handle_conv2d(f, op_type)
        elif "decode" in op_type:
            return self.handle_decode_matmul(f, op_type)
        elif "matmul" in op_type:
            return self.handle_matmul(f, op_type)
        elif "attention" in op_type:
            return self.handle_attention(f, op_type)
        elif "layer_norm" in op_type or "rms_norm" in op_type:
            return self.handle_norm(f, op_type)

        raise ValueError("Unsupported composite {}".format(op_type))

    def visit_span(self, span):
        return span


@register_func("contrib.cutlass.tune_relax_function")
def profile_relax_function(functions, options):
    """Tune and annotate CUTLASS composite functions with shape, dtype and generated templates."""
    tmp_dir = options.get("tmp_dir", "./tmp")
    sm = options.get("sm", 80)
    conv2d_profiler = CutlassConv2DProfiler(sm, _get_cutlass_path(), tmp_dir)
    gemm_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), tmp_dir)

    annotated_functions = []

    for f in functions:
        annotator = CutlassRelaxFunctionAnnotator(
            tvm.IRModule.from_expr(f), conv2d_profiler, gemm_profiler, options
        )
        annotated_functions.append(annotator.visit_expr(f))

    return annotated_functions


@register_func("contrib.cutlass.compile")
def compile_cutlass_module(c_source_module, options):
    """Compile all CUTLASS kernels in the given C-source module.

    Parameters
    ----------
    c_source_module: runtime.Module
        A C-source module containing CUTLASS kernels.

    options: dict
        Compilation options. Currently recognizes
          "sm": The target architecture (compute capability), for example 75 or 80 (default: 80)
          "threads": The number of threads to use in NVCC parallel compilation (default:
          use all logical cores)
          "use_fast_math": Whether or not to use faster but approximate arithmetic in some
          CUTLASS epilogues (default: False)

    Returns
    -------
    rt_mod : runtime.Module
        A runtime module where all cutlass kernels have been compiled.
    """
    tmp_dir = options.get("tmp_dir", "./tmp")
    defaults = {"sm": 80, "threads": -1, "use_fast_math": False}
    compile_config = {key: options.get(key, val) for key, val in defaults.items()}

    function_names = c_source_module.get_function("get_func_names")()
    compile_options = _get_cutlass_compile_options(**compile_config)
    lib_path = os.path.join(tmp_dir, "cutlass.o")
    logger.info("Compiling generated CUTLASS code")
    c_source_module.export_library(lib_path, workspace_dir=tmp_dir, **compile_options)

    # Recover static library
    return tvm.runtime.load_static_library(lib_path, function_names)


@register_func("relay.ext.cutlass.compile_for_cutlass")
def compile_for_cutlass(mod, cutlass_target):
    """Given an IRModule with at least one Compiler='cutlass' Relay function, return a
    LibraryModule with all such functions compiled into their PackedFunc-compatible form.
     - First runs CUTLASS tuning to decide on the best kernels, which itself requires the
       repeated compilation and execution of CUDA code using nvcc. The results of this
       is captured as annotation on each relevant function. Kernel performance is cached
       overall all functions.
     - Then generates a single CSourceModule containing C code implementing all the
       Compiler='cutlass' Relay functions, accounting for the tuning done above.
     - Then compiles that CSourceModule with the appropriate nvcc arguments to yield
       a static .o library. An export_library step will be required on the final runtime
       module to link that library into the overall .so library.
     See CompileForCutlass in src/relay/backend/contrib/cutlass/codegen.cc for where this
     helper function is used to implement the RelayToTIR pass hook for CUTLASS."""

    # Recover options from the current 'cutlass' Target
    assert cutlass_target.kind.name == "cutlass"
    tuning_config = {
        key: cutlass_target.attrs.get(key)
        for key in [
            "sm",
            "use_3xtf32",
            "split_k_slices",
            "profile_all_alignments",
            "find_first_valid",
            "use_multiprocessing",
        ]
    }
    compile_config = {
        key: cutlass_target.attrs.get(key) for key in ["sm", "threads", "use_fast_math"]
    }
    tmp_dir = cutlass_target.attrs.get("tmp_dir")
    compile_config["tmp_dir"] = tmp_dir

    # Tune
    logger.info("Tuning for CUTLASS")
    mod, _ = tune_cutlass_kernels(mod, tmp_dir=tmp_dir, **tuning_config)

    # Compile
    logger.info("Creating CSource module for CUTLASS")
    create_c_source_module = tvm._ffi.get_global_func("relay.ext.cutlass.create_c_source_module")
    c_module = create_c_source_module(mod)
    return compile_cutlass_module(c_module, compile_config)


def finalize_modules(lib, lib_path="compile.so", tmp_dir="./tmp"):
    """Returns lib with any C source, LLVM and static library modules complied and linked in ready
    for use by the graph or AOT executors. This method is not specific to CUTLASS, however it does
    assume nvcc will be used for final compilation and linking. It is provided here for
    convenience.

    Parameters
    ----------
    lib : runtime.Module
        The output from relay.build.

    lib_path : string
        The path to a shared library which will be generated as the result of the build process.

    tmp_dir : string
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    updated_lib : runtime.Module
        The updated library with all compilation and linking completed.

    """
    lib_path = os.path.join(tmp_dir, lib_path)
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    return runtime.load_module(lib_path)


def finalize_modules_vm(vm_exec, lib_path="compile.so", vmcode_path="vmcode.ro", tmp_dir="./tmp"):
    """Returns vm_exec with any C source, LLVM and static library modules compiled and linked in
    ready for use by the VM executor. This method is not specific to CUTLASS, however it does
    assume nvcc will be used for final compilation and linking. It is provided here for
    convenience.

    Parameters
    ----------
    vm_exec : vm.Executable
        The output from relay.vm.compile containing compiled host code and kernels.

    lib_path : string
        The path to a shared library which will be generated as the result of the build process.

    vmcode_path : string
        The path where the VM bytecode will be serialized to as a side-effect.

    tmp_dir : string
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    updated_vm_exec : vm.Executable
        The updated VM executable with all compilation and linking completed.
    """
    code, lib = vm_exec.save()
    lib_path = os.path.join(tmp_dir, lib_path)
    vmcode_path = os.path.join(tmp_dir, vmcode_path)
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    with open(vmcode_path, "wb") as fo:
        fo.write(code)
    lib = tvm.runtime.load_module(lib_path)
    return tvm.runtime.vm.Executable.load_exec(code, lib)
