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
# pylint: disable=invalid-name, unused-argument, logging-format-interpolation
"""TensorRT supported operators."""
import logging
from typing import Tuple, List, Dict, Union, Optional, Any, Callable

import numpy as np  # type: ignore
import tvm
from tvm import relay
from tvm.ir import Op
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant, is_tuple, is_tuple_get_item
from tvm.relay.expr import Call, Constant, TupleGetItem
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.op.contrib.register import register_pattern_table

logger = logging.getLogger("TensorRT")


def is_tensorrt_compiler_enabled() -> bool:
    return tvm.get_global_func("relay.ext.tensorrt.is_runtime_enabled", True) is not None


def is_tensorrt_runtime_enabled() -> bool:
    """Check if the TensorRT graph executor is present.
    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.ext.tensorrt.is_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def get_tensorrt_target() -> tvm.target.Target:
    """Returns the current Target, which must be of kind "tensorrt"."""
    target = tvm.target.Target.current()
    if target is None or target.kind.name != "tensorrt":
        # Create the default target.
        return tvm.target.Target("tensorrt")
    return target


def get_tensorrt_version() -> Tuple[int, int, int]:
    """Returns the version of TensorRT to assume during compilation.
    In order of preference this is taken from:
     - The current "tensorrt" target's "tensorrt_version" attribute string.
     - The version linked to the TVM runtime.
     - (6, 0, 1)

    Returns
    -------
    ret: Tuple[int, int, int]
        TensorRT version as a tuple of (major, minor, patch).
    """
    # cf logic in tensorrt/codegen.cc::SaveGlobalAttributes
    # First check for version in target.
    target = get_tensorrt_target()
    version = target.attrs["tensorrt_version"]
    if len(version) == 3:
        return int(version[0]), int(version[1]), int(version[2])
    assert len(version) == 0

    # Next, ask runtime for its version.
    if is_tensorrt_runtime_enabled():
        get_version = tvm.get_global_func("relay.ext.tensorrt.get_version")
        version = get_version()
        assert len(version) == 3
        return int(version[0]), int(version[1]), int(version[2])

    # Finally, use default.
    logger.warning(
        "TVM was not built against TensorRT and no version was provided in the 'tensorrt' target."
        "Defaulting to 6.0.1."
    )
    return (6, 0, 1)


def get_tensorrt_use_implicit_batch_mode() -> bool:
    """Returns the "use_implicit_batch" attribute of the current "tensorrt" target."""
    target = get_tensorrt_target()
    return target.attrs["use_implicit_batch"]


def get_tensorrt_remove_no_mac_subgraphs() -> bool:
    """Returns the "remove_no_mac_subgraphs" attribute of the current "tensorrt" target."""
    target = get_tensorrt_target()
    return target.attrs["remove_no_mac_subgraphs"]


def get_tensorrt_use_fp16() -> bool:
    """Returns the "use_fp16" attribute of the current "tensorrt" target."""
    target = get_tensorrt_target()
    return target.attrs["use_fp16"]


def partition_for_tensorrt(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.NDArray]] = None,
    # CAUTION: Can't use default Target("tensorrt") here since the target kind is only available
    #          if is_tensorrt_compiler_enabled() == True.
    target: Optional[tvm.target.Target] = None,
) -> tvm.IRModule:
    """Partition all functions in mod to greedily offload supported operators to TensorRT.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to partition.
    target : tvm.target.Target
        A target of kind "tensorrt" describing additional partitioning and compilation options.
    params : Optional[Dict[str, tvm.nd.NDArray]]
        Constant input parameters.

    Returns
    -------
    partitioned_mod : tvm.IRModule
        The partitioned module.

    """
    assert is_tensorrt_compiler_enabled(), "Can only partition for TensorRT if it is enabled"
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    if target is None:
        # Use a default target. The get_tensorrt_target() function will similarly create an
        # equivalent default target when compilation continues after partitioning.
        target = tvm.target.Target("tensorrt")

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            RemoveDropoutPass(),
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(
                {
                    "nn.conv1d": ["NCW", "default"],
                    "nn.conv2d": ["NCHW", "default"],
                    "nn.conv3d": ["NCDHW", "default"],
                    "nn.conv2d_transpose": ["NCHW", "default"],
                }
            ),
            transform.FoldConstant(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    with target:
        mod = seq(mod)
        mod = prune_tensorrt_subgraphs(mod)
    return mod


def is_supported_trt_type(typ: Union[tvm.ir.TensorType, tvm.ir.TupleType], op_name: str) -> bool:
    """Check whether a type is supported by TensorRT."""
    supported_dtypes = ["float32"]
    if get_tensorrt_use_fp16():
        supported_dtypes.append("float16")
    if isinstance(typ, tvm.ir.TensorType):
        if typ.dtype not in supported_dtypes:
            logger.info(f"{op_name}: Only {supported_dtypes} tensor dtypes are supported.")
            return False
        dims = typ.shape
        if get_tensorrt_use_implicit_batch_mode():
            # The first dimension can be Any.
            dims = dims[1:]
        for dim in dims:
            if isinstance(dim, tvm.tir.expr.Any):
                logger.info(f"{op_name}: Only statically known tensor shapes are supported.")
                return False
    elif isinstance(typ, tvm.ir.TupleType):
        for field_type in typ.fields:
            if not is_supported_trt_type(field_type, op_name):
                return False
    else:
        logger.info(f"{op_name}: Type {typ} is not supported.")
        return False
    return True


def get_op_name(expr: relay.expr.Expr) -> str:
    """Get the operator name from an expression."""
    if isinstance(expr, Op):
        return expr.name
    if isinstance(expr, Call):
        return get_op_name(expr.op)
    if isinstance(expr, TupleGetItem):
        return get_op_name(expr.tuple_value)
    if isinstance(expr, relay.Tuple):
        return get_op_name(expr.fields[0])
    return ""


def get_args(expr: relay.expr.Expr) -> List[relay.expr.Expr]:
    """Get the arguments from an expression."""
    if isinstance(expr, Call):
        return expr.args
    if isinstance(expr, TupleGetItem):
        return get_args(expr.tuple_value)
    if isinstance(expr, relay.Tuple):
        return [arg for args in map(get_args, expr.fields) for arg in args]
    return []


def get_attrs(expr: relay.expr.Expr) -> Any:
    """Get the attributes from an expression."""
    if isinstance(expr, Call):
        return expr.attrs
    if isinstance(expr, TupleGetItem):
        return get_attrs(expr.tuple_value)
    return {}


CheckFunc = Callable[[Any, List[relay.expr.Expr], str], bool]


def make_predicate(checker: CheckFunc) -> Callable[[relay.expr.Expr], bool]:
    """Returns the pattern predicate which performs the standard checks, then invokes the
    more primitive checker."""

    def predicate(expr: relay.expr.Expr) -> bool:
        op_name = get_op_name(expr)
        attrs = get_attrs(expr)
        args = get_args(expr)
        if not all([is_supported_trt_type(arg.checked_type, op_name) for arg in args]):
            return False
        if not checker(attrs, args, op_name):
            return False
        logger.info(f"{op_name}: Predicate passes")
        return True

    return predicate


standard_predicate = make_predicate(lambda attrs, args, op_name: True)


def make_trt_version_checker(version: Tuple[int, int, int]) -> CheckFunc:
    """Helper for ops which require a minimum TRT version"""

    def checker(attrs: Any, args: List[relay.expr.Expr], op_name: str) -> bool:
        if get_tensorrt_version() < version:
            logger.info(
                f"{op_name}: requires TensorRT version {'.'.join(map(str, version))} or higher."
            )
            return False
        return True

    return checker


def make_and_checker(*checkers: CheckFunc) -> CheckFunc:
    def checker(attrs: Any, args: List[relay.expr.Expr], op_name: str) -> bool:
        return all([c(attrs, args, op_name) for c in checkers])

    return checker


def multiply_checker(attrs: Any, args: List[relay.expr.Expr], op_name: str) -> bool:
    """Helper for multiply operations."""
    shapes = [
        [int(x) if not isinstance(x, tvm.tir.expr.Any) else -1 for x in arg.checked_type.shape]
        for arg in args
    ]
    # TODO(mbs): Follow up
    # Batched multiply operations don't work in implicit batch mode. The following shapes
    # have been excluded because they occur in PT MaskRCNN model. The long term solution is
    # to switch to explicit batch mode after performance regressions are solved.
    if all([list(map(int, shape)) in [[300, 64, 7, 7], [300, 1, 1, 1]] for shape in shapes]):
        logger.info(f"{op_name}: Excluding since problematic in implicit batch mode")
        return False
    return True


def reduce_checker(attrs: Any, args: List[relay.expr.Expr], op_name: str) -> bool:
    """Helper for reduce operations."""
    if get_tensorrt_use_implicit_batch_mode() and (not attrs.axis or len(attrs.axis) == 0):
        logger.info(f"{op_name}: cannot reduce to scalar.")
        return False
    if attrs.exclude:
        logger.info(f"{op_name}: exclude not supported.")
        return False
    if get_tensorrt_use_implicit_batch_mode() and any([x == 0 for x in map(int, attrs.axis)]):
        logger.info(f"{op_name}: can't modify batch dimension.")
        return False
    return True


def add_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if add is supported by TensorRT."""
    shapes = [
        [int(x) if not isinstance(x, tvm.tir.expr.Any) else -1 for x in arg.checked_type.shape]
        for arg in args
    ]

    # Scalars require explicit batch mode.
    if get_tensorrt_use_implicit_batch_mode() and any([len(shape) < 1 for shape in shapes]):
        logger.info(f"{op_name}: Scalars not supported in implicit batch mode")
        return False

    if (
        not get_tensorrt_use_implicit_batch_mode()
        and (isinstance(args[0], Constant) or isinstance(args[1], Constant))
        and len(shapes[0]) > 0
        and len(shapes[1]) > 0
        and shapes[0][0] == shapes[1][0]
        and shapes[0][0] != 1
        and (len(shapes[0]) > 3 or len(shapes[1]) > 3)
    ):
        logger.info(f"{op_name}: bug in TRT with adding batched constants.")
        return False
    return True


def batch_norm_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.batch_norm is supported by TensorRT."""
    if len(args[0].checked_type.shape) == 5 and get_tensorrt_version() < (6, 0, 1):
        logger.info(f"{op_name}: TensorRT 6.0.1 or higher is required for rank 5 inputs.")
        return False
    if len(args[0].checked_type.shape) > 5:
        logger.info(f"{op_name}: Input rank must be 5 or less.")
        return False
    if int(attrs.axis) not in (1, 3):
        logger.info(f"{op_name}: axis is {int(attrs.axis)} but must be 1 or 3.")
        return False
    return True


def softmax_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.softmax is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axis) == 0:
        logger.info(f"{op_name}: can't modify batch dimension.")
        return False
    return True


def conv1d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.conv1d is supported by TensorRT."""
    if not isinstance(args[1], Constant):
        logger.info(f"{op_name}: kernel argument must be constant.")
        return False
    if attrs.data_layout != "NCW":
        logger.info(f"{op_name}: data_layout is {attrs.data_layout} but must be NCW.")
        return False
    if attrs.kernel_layout != "OIW":
        logger.info(f"{op_name}: kernel_layout is {attrs.kernel_layout} but must be OIW.")
        return False
    return True


def conv2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.conv2d is supported by TensorRT."""
    assert len(args) == 2
    if not isinstance(args[1], Constant):
        logger.info(f"{op_name}: kernel argument must be constant.")
        return False
    if attrs.data_layout != "NCHW":
        logger.info(f"{op_name}: data_layout is {attrs.data_layout} but must be NCHW.")
        return False
    if attrs.kernel_layout != "OIHW":
        logger.info(f"{op_name}: kernel_layout is {attrs.kernel_layout} but must be OIHW.")
        return False
    if attrs.out_layout and attrs.out_layout != "NCHW":
        logger.info(f"{op_name}: out_layout is {attrs.out_layout} but must be NCHW.")
        return False
    return True


def dense_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if dense is supported by TensorRT."""
    if not isinstance(args[1], Constant):
        logger.info(f"{op_name}: weight must be constant")
        return False
    input_rank = len(args[0].checked_type.shape)
    weight_rank = len(args[1].checked_type.shape)
    if input_rank not in (2, 3, 4):
        logger.info(f"{op_name}: input has rank {input_rank} but must be 2, 3 or 4.")
        return False
    if weight_rank != 2:
        logger.info(f"{op_name}: weight has rank {weight_rank} but must be 2.")
        return False
    return True


def batch_matmul_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if dense is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode() and len(args[0].checked_type.shape) != len(
        args[1].checked_type.shape
    ):
        logger.info(f"{op_name}: requires use_implict_batch=False.")
        return False
    return True


def layer_norm_checker(attrs: Any, args: List[relay.expr.Expr], op_name: str) -> bool:
    """Check if dense is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axis) == 0:
        logger.info(f"{op_name}: requires use_implict_batch=False.")
        return False
    return True


def bias_add_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.bias_add is supported by TensorRT."""
    input_rank = len(args[0].checked_type.shape)
    if input_rank not in (2, 3, 4):
        logger.info(f"{op_name}: input rank is {input_rank} but must be 2, 3 or 4.")
        return False
    return True


def max_pool_2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.max_pool2d is supported by TensorRT."""
    if attrs.layout != "NCHW":
        logger.info(f"{op_name}: layout is {attrs.layout} but must be NCHW.")
        return False
    if attrs.ceil_mode and get_tensorrt_version() < (5, 1, 5):
        logger.info(f"{op_name}: ceil_mode=True requires TensorRT 5.1.5 or greater.")
        return False
    return True


def avg_pool_2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.avg_pool2d is supported by TensorRT."""
    if attrs.layout != "NCHW":
        logger.info(f"{op_name}: layout is {attrs.layout} but must be NCHW.")
        return False
    if (
        attrs.count_include_pad
        and len(attrs.padding) == 4
        and (
            int(attrs.padding[0]) != int(attrs.padding[2])
            or int(attrs.padding[1]) != int(attrs.padding[3])
        )
    ):
        logger.info(
            f"{op_name}: inclusive-counted blended or average "
            "pooling is not supported in combination with asymmetric padding"
        )
        return False
    if attrs.ceil_mode and get_tensorrt_version() < (5, 1, 5):
        logger.info(f"{op_name}: ceil_mode=True requires TensorRT 5.1.5 or greater.")
        return False
    return True


def global_max_pool_2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.global_max_pool2d is supported by TensorRT."""
    if attrs.layout != "NCHW":
        logger.info(f"{op_name}: layout is {attrs.layout} but must be NCHW.")
        return False
    return True


def global_avg_pool_2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.global_avg_pool2d is supported by TensorRT."""
    if attrs.layout != "NCHW":
        logger.info(f"{op_name}: layout is {attrs.layout} but must be NCHW.")
        return False
    return True


def expand_dims_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if expand_dims is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axis) == 0:
        logger.info(f"{op_name}: can't modify batch dimension.")
        return False
    return True


def squeeze_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if squeeze is supported by TensorRT."""
    if not attrs.axis:
        logger.info(f"{op_name}: must explicitly set axis.")
        return False
    if get_tensorrt_use_implicit_batch_mode() and any([axis == 0 for axis in map(int, attrs.axis)]):
        logger.info(f"{op_name}: can't modify batch dimension.")
        return False
    return True


def concatenate_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if concatenate is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode():
        if int(attrs.axis) == 0:
            logger.info(f"{op_name}: can't modify batch dimension.")
            return False

    if not isinstance(args[0], relay.Tuple):
        logger.info("f{op_name}: concatenate must be applied to a literal tuple")
        return False

    for tuple_input in args[0].fields:
        if isinstance(tuple_input, Constant):
            logger.info(f"{op_name}: can't concatenate tensors with constants.")
            return False

    return True


def split_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if split is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axis) == 0:
        logger.info(f"{op_name}: can't modify batch dimension.")
        return False
    return True


def conv2d_transpose_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.conv2d_transpose is supported by TensorRT."""
    if attrs.data_layout != "NCHW":
        logger.info(f"{op_name}: data_layout is {attrs.data_layout} but must be NCHW.")
        return False
    if attrs.kernel_layout != "OIHW":
        logger.info(f"{op_name}: kernel_layout is {attrs.kernel_layout} but must be OIHW.")
        return False
    if attrs.out_layout and attrs.out_layout != "NCHW":
        logger.info(f"{op_name}: out_layout is {attrs.out_layout} but must be NCHW.")
        return False
    if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
        logger.info(f"{op_name}: dilation rate must be 1.")
        return False
    return True


def transpose_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if transpose is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axes[0]) != 0:
        logger.info(f"{op_name}: can't modify batch dimension.")
        return False
    return True


def layout_transform_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if layout_transform is supported by TensorRT."""
    if (attrs.src_layout, attrs.dst_layout) not in [
        ("NCHW", "NHWC"),
        ("NHWC", "NCHW"),
        ("NDHWC", "NCDHW"),
        ("NCDHW", "NDHWC"),
    ]:
        logger.info(f"{op_name}: {attrs.src_layout} to {attrs.dst_layout} is not supported.")
        return False
    return True


def reshape_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if reshape is supported by TensorRT."""
    if any([x < -1 for x in map(int, attrs.newshape)]):
        logger.info(f"{op_name}: new shape dims must be explicit.")
        return False
    if get_tensorrt_use_implicit_batch_mode():
        shape = args[0].checked_type.shape
        new_shape = attrs.newshape
        if len(new_shape) == 0 or len(shape) == 0:
            logger.info(f"{op_name}: Can't reshape to or from scalar.")
            return False
        dynamic_reshape = any([isinstance(x, tvm.tir.expr.Any) for x in shape])

        if dynamic_reshape:
            # Make sure that the batch dim is unmodified.
            if int(new_shape[0]) < 0:
                for shape_val, new_shape_val in zip(shape[1:], new_shape[1:]):
                    if not (
                        isinstance(shape_val, (int, tvm.tir.expr.IntImm))
                        and isinstance(new_shape_val, (int, tvm.tir.expr.IntImm))
                        and int(shape_val) == int(new_shape_val)
                    ):
                        logger.info(f"{op_name}: can't modify batch dimension")
                        return False
            elif int(new_shape[0]) > 0:
                # Currently we only allow dim[0] to be Any, so this branch will always be False
                if not (
                    isinstance(shape[0], (int, tvm.tir.expr.IntImm))
                    and isinstance(new_shape[0], (int, tvm.tir.expr.IntImm))
                    and int(shape[0]) == int(new_shape[0])
                ):
                    logger.info(f"{op_name}: can't modify batch dimension")
                    return False
        else:
            shape = list(map(int, shape))
            new_shape = list(map(int, new_shape))

            # TRT cannot modify batch dimension.
            original_volume = np.prod(shape)
            # First, resolve 0.
            for i, value in enumerate(new_shape):
                if value == 0:
                    new_shape[i] = shape[i]
            # Resolve -1.
            for i, value in enumerate(new_shape):
                if value == -1:
                    new_shape[i] = original_volume // np.prod([x for x in new_shape if x != -1])
            # Remove batch dimension and see if volumes match
            if shape[0] != new_shape[0]:
                logger.info(f"{op_name}: can't modify batch dimension.")
                return False
    return True


def pad_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.pad is supported by TensorRT."""
    pad_value = args[1]
    if not isinstance(pad_value, relay.Constant):
        logger.info(f"{op_name}: pad argument must be constant")
        return False
    pad_value = pad_value.data.numpy().item()
    if attrs.pad_mode != "constant":
        logger.info(f"{op_name}: pad mode is {attrs.pad_mode} but must be constant.")
        return False
    if pad_value > 0.0:
        logger.info(f"{op_name}: pad value is {pad_value} but must be 0.0.")
        return False
    if len(attrs.pad_width) not in [4, 5]:
        logger.info(f"{op_name}: can only pad 4D or 5D inputs")
        return False
    if any([x != 0 for x in attrs.pad_width[0]]) or any([x != 0 for x in attrs.pad_width[1]]):
        logger.info(f"{op_name}: can't pad batch or channel dimensions.")
        return False
    if len(attrs.pad_width) == 5 and any([x != 0 for x in attrs.pad_width[2]]):
        logger.info(f"{op_name}: can only pad last two dimensions for 5D inputs.")
        return False
    return True


def strided_slice_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if strided_slice is supported by TensorRT."""
    if get_tensorrt_use_implicit_batch_mode():
        batch_dim_begin_modified = attrs.begin[0] is not None and int(attrs.begin[0]) != 0
        batch_dim_end_modified = (
            attrs.end[0] is not None
            and int(attrs.end[0]) != -1
            and int(attrs.end[0]) != int(args[0].checked_type.shape[0])
        )
        if batch_dim_begin_modified or batch_dim_end_modified:
            logger.info(f"{op_name}: can't modify batch dimension.")
            return False
    if any([x is not None and x <= 0 for x in attrs.strides]):
        logger.info(f"{op_name}: stride must be positive")
        return False
    for i in range(0, len(args[0].checked_type.shape)):
        begin = int(attrs.begin[i])
        if attrs.slice_mode == "end":
            end = (
                int(attrs.end[i])
                if attrs.end[i] is not None and int(attrs.end[i]) != -1
                else args[0].checked_type.shape[i]
            )
            size = int(end) - int(begin)
        elif attrs.slice_mode == "size":
            size = (
                int(attrs.end[i])
                if attrs.end[i] is not None and int(attrs.end[i]) != -1
                else args[0].checked_type.shape[i] - begin
            )
        else:
            logger.warning(f"{op_name}: unknown slice mode encountered")
            size = 1

        if int(size) < 1:
            logger.info(f"{op_name}: size of slice must be at least 1")
            return False

    return True


def adaptive_max_pool2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.adaptive_max_pool2d is supported by TensorRT."""
    if len(attrs.output_size) == 0 or any([size != 1 for size in map(int, attrs.output_size)]):
        logger.info(f"{op_name}: output size must be (1, 1).")
        return False
    return True


def adaptive_avg_pool2d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.adaptive_avg_pool2d is supported by TensorRT."""
    if len(attrs.output_size) == 0 or any([size != 1 for size in map(int, attrs.output_size)]):
        logger.info(f"{op_name}: output size must be (1, 1).")
        return False
    return True


def conv3d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.conv3d is supported by TensorRT."""
    if not isinstance(args[1], Constant):
        logger.info(f"{op_name}: kernel argument must be constant.")
        return False
    if attrs.data_layout != "NCDHW":
        logger.info(f"{op_name}: data_layout is {attrs.data_layout} but must be NCDHW.")
        return False
    if attrs.kernel_layout != "OIDHW":
        logger.info(f"{op_name}: kernel_layout is {attrs.kernel_layout} but must be OIDHW.")
        return False
    if attrs.out_layout and attrs.out_layout != "NCDHW":
        logger.info(f"{op_name}: out_layout is {attrs.out_layout} but must be NCDHW.")
        return False
    return True


def max_pool_3d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.max_pool3d is supported by TensorRT."""
    if attrs.layout != "NCDHW":
        logger.info(f"{op_name}: layout is {attrs.layout} but must be NCDHW.")
        return False
    return True


def avg_pool_3d_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.avg_pool3d is supported by TensorRT."""
    if attrs.layout != "NCDHW":
        logger.info(f"{op_name}: layout is {attrs.layout} but must be NCDHW.")
        return False
    return True


def conv3d_transpose_checker(
    attrs: Any, args: List[relay.expr.Expr], op_name: str
) -> bool:  # pylint: disable=unused-variable
    """Check if nn.conv3d_transpose is supported by TensorRT."""
    if attrs.data_layout != "NCDHW":
        logger.info(f"{op_name}: data_layout is {attrs.data_layout} but must be NCDHW.")
        return False
    if attrs.kernel_layout != "OIDHW":
        logger.info(f"{op_name}: kernel_layout is {attrs.kernel_layout} but must be OIDHW.")
        return False
    if attrs.out_layout and attrs.out_layout != "NCDHW":
        logger.info(f"{op_name}: out_layout is {attrs.out_layout} but must be NCDHW.")
        return False
    if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
        logger.info(f"{op_name}: dilation rate must be 1.")
        return False
    if attrs.output_padding and any([x != 0 for x in map(int, attrs.output_padding)]):
        logger.info(f"{op_name}: output padding is not supported.")
        return False
    return True


def unary_op_pattern(op: relay.expr.Expr) -> relay.dataflow_pattern.DFPattern:
    """Matches unary operation"""
    return is_op(op)(wildcard())


def unary_op_pattern_with_any_tuple(op: relay.expr.Expr) -> relay.dataflow_pattern.DFPattern:
    """Matches unary operation with literal tuple argument"""
    return is_op(op)(is_tuple(None))


def binary_op_pattern(op: relay.expr.Expr) -> relay.dataflow_pattern.DFPattern:
    """Matches binary operation"""
    return is_op(op)(wildcard(), wildcard())


def binary_op_pattern_with_const(op: relay.expr.Expr) -> relay.dataflow_pattern.DFPattern:
    """Matches binary operation with rhs arg a constant"""
    return is_op(op)(wildcard(), is_constant())


def proj_five_op_pattern_with_const(op: relay.expr.Expr) -> relay.dataflow_pattern.DFPattern:
    return is_tuple_get_item(
        is_op(op)(wildcard(), is_constant(), is_constant(), is_constant(), is_constant()), 0
    )


@register_pattern_table("tensorrt")
def pattern_table() -> List[
    Tuple[str, relay.dataflow_pattern.DFPattern, Callable[[relay.expr.Call], bool]]
]:
    """Get the Tensorrt compiler pattern table for supported ops."""

    return [
        (
            "tensorrt.nn.conv3d",
            binary_op_pattern_with_const("nn.conv3d"),
            make_predicate(make_and_checker(make_trt_version_checker((6, 0, 1)), conv3d_checker)),
        ),
        (
            "tensorrt.nn.conv2d",
            binary_op_pattern_with_const("nn.conv2d"),
            make_predicate(conv2d_checker),
        ),
        (
            "tensorrt.nn.conv1d",
            binary_op_pattern_with_const("nn.conv1d"),
            make_predicate(conv1d_checker),
        ),
        (
            "tensorrt.nn.conv2d_transpose",
            binary_op_pattern("nn.conv2d_transpose"),
            make_predicate(conv2d_transpose_checker),
        ),
        ("tensorrt.squeeze", binary_op_pattern("squeeze"), make_predicate(squeeze_checker)),
        ("tensorrt.add", binary_op_pattern("add"), make_predicate(add_checker)),
        (
            "tensorrt.nn.dense",
            binary_op_pattern_with_const("nn.dense"),
            make_predicate(dense_checker),
        ),
        (
            "tensorrt.nn.bias_add",
            binary_op_pattern("nn.bias_add"),
            make_predicate(bias_add_checker),
        ),
        (
            "tensorrt.nn.batch_matmul",
            binary_op_pattern("nn.batch_matmul"),
            make_predicate(batch_matmul_checker),
        ),
        ("tensorrt.divide", binary_op_pattern("divide"), standard_predicate),
        ("tensorrt.multiply", binary_op_pattern("multiply"), make_predicate(multiply_checker)),
        ("tensorrt.subtract", binary_op_pattern("subtract"), standard_predicate),
        ("tensorrt.power", binary_op_pattern("power"), standard_predicate),
        ("tensorrt.maximum", binary_op_pattern("maximum"), standard_predicate),
        ("tensorrt.minimum", binary_op_pattern("minimum"), standard_predicate),
        ("tensorrt.nn.relu", unary_op_pattern("nn.relu"), standard_predicate),
        (
            "tensorrt.nn.leaky_relu",
            unary_op_pattern("nn.leaky_relu"),
            make_predicate(make_trt_version_checker((5, 1, 5))),
        ),
        ("tensorrt.nn.pad", unary_op_pattern("nn.pad"), standard_predicate),
        ("tensorrt.sigmoid", unary_op_pattern("sigmoid"), standard_predicate),
        ("tensorrt.tanh", unary_op_pattern("tanh"), standard_predicate),
        ("tensorrt.exp", unary_op_pattern("exp"), standard_predicate),
        ("tensorrt.log", unary_op_pattern("log"), standard_predicate),
        ("tensorrt.sqrt", unary_op_pattern("sqrt"), standard_predicate),
        ("tensorrt.abs", unary_op_pattern("abs"), standard_predicate),
        ("tensorrt.negative", unary_op_pattern("negative"), standard_predicate),
        ("tensorrt.nn.batch_flatten", unary_op_pattern("nn.batch_flatten"), standard_predicate),
        ("tensorrt.clip", unary_op_pattern("clip"), standard_predicate),
        (
            "tensorrt.sin",
            unary_op_pattern("sin"),
            make_predicate(make_trt_version_checker((5, 1, 5))),
        ),
        (
            "tensorrt.cos",
            unary_op_pattern("cos"),
            make_predicate(make_trt_version_checker((5, 1, 5))),
        ),
        (
            "tensorrt.atan",
            unary_op_pattern("atan"),
            make_predicate(make_trt_version_checker((5, 1, 5))),
        ),
        (
            "tensorrt.ceil",
            unary_op_pattern("ceil"),
            make_predicate(make_trt_version_checker((5, 1, 5))),
        ),
        ("tensorrt.floor", unary_op_pattern("floor"), standard_predicate),
        (
            "tensorrt.erf",
            unary_op_pattern("erf"),
            make_predicate(make_trt_version_checker((7, 0, 0))),
        ),
        ("tensorrt.sum", unary_op_pattern("sum"), make_predicate(reduce_checker)),
        ("tensorrt.prod", unary_op_pattern("prod"), make_predicate(reduce_checker)),
        ("tensorrt.max", unary_op_pattern("max"), make_predicate(reduce_checker)),
        ("tensorrt.min", unary_op_pattern("min"), make_predicate(reduce_checker)),
        ("tensorrt.max", unary_op_pattern("max"), make_predicate(reduce_checker)),
        ("tensorrt.mean", unary_op_pattern("mean"), make_predicate(reduce_checker)),
        (
            "tensorrt.concatenate",
            unary_op_pattern_with_any_tuple("concatenate"),
            make_predicate(concatenate_checker),
        ),
        (
            "tensorrt.expand_dims",
            unary_op_pattern("expand_dims"),
            make_predicate(expand_dims_checker),
        ),
        (
            "tensorrt.layout_transform",
            unary_op_pattern("layout_transform"),
            make_predicate(layout_transform_checker),
        ),
        ("tensorrt.transpose", unary_op_pattern("transpose"), make_predicate(transpose_checker)),
        ("tensorrt.reshape", unary_op_pattern("reshape"), make_predicate(reshape_checker)),
        ("tensorrt.split", unary_op_pattern("split"), make_predicate(split_checker)),
        ("tensorrt.nn.pad", unary_op_pattern("nn.pad"), make_predicate(pad_checker)),
        (
            "tensorrt.strided_slice",
            unary_op_pattern("strided_slice"),
            make_predicate(
                make_and_checker(make_trt_version_checker((5, 1, 5)), strided_slice_checker)
            ),
        ),
        (
            "tensorrt.nn.adaptive_avg_pool2d",
            unary_op_pattern("nn.adaptive_avg_pool2d"),
            make_predicate(adaptive_avg_pool2d_checker),
        ),
        (
            "tensorrt.nn.adaptive_max_pool2d",
            unary_op_pattern("nn.adaptive_max_pool2d"),
            make_predicate(adaptive_max_pool2d_checker),
        ),
        (
            "tensorrt.nn.max_pool3d",
            unary_op_pattern("nn.max_pool3d"),
            make_predicate(
                make_and_checker(make_trt_version_checker((6, 0, 1)), max_pool_3d_checker)
            ),
        ),
        (
            "tensorrt.nn.avg_pool3d",
            unary_op_pattern("nn.avg_pool3d"),
            make_predicate(
                make_and_checker(make_trt_version_checker((6, 0, 1)), avg_pool_3d_checker)
            ),
        ),
        (
            "tensorrt.nn.conv3d_transpose",
            unary_op_pattern("nn.conv3d_transpose"),
            make_predicate(
                make_and_checker(make_trt_version_checker((6, 0, 1)), conv3d_transpose_checker)
            ),
        ),
        ("tensorrt.nn.softmax", unary_op_pattern("nn.softmax"), make_predicate(softmax_checker)),
        (
            "tensorrt.nn.layer_norm",
            unary_op_pattern("nn.layer_norm"),
            make_predicate(layer_norm_checker),
        ),
        (
            "tensorrt.nn.max_pool2d",
            unary_op_pattern("nn.max_pool2d"),
            make_predicate(max_pool_2d_checker),
        ),
        (
            "tensorrt.nn.avg_pool2d",
            unary_op_pattern("nn.avg_pool2d"),
            make_predicate(avg_pool_2d_checker),
        ),
        (
            "tensorrt.nn.global_max_pool2d",
            unary_op_pattern("nn.global_max_pool2d"),
            make_predicate(global_max_pool_2d_checker),
        ),
        (
            "tensorrt.nn.global_avg_pool2d",
            unary_op_pattern("nn.global_avg_pool2d"),
            make_predicate(global_avg_pool_2d_checker),
        ),
        (
            "tensorrt.nn.batch_norm",
            proj_five_op_pattern_with_const("nn.batch_norm"),
            make_predicate(batch_norm_checker),
        ),
    ]


class IsComputeIntensiveGraph(ExprVisitor):
    """
    Visits the Graph recursively and checks if it contains compute heavy ops like convolutions and
    its transpose, dense and batch mat-mul.
    """

    def __init__(self) -> None:
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call: relay.expr.Call) -> None:
        compute_intensive_ops = {
            "nn.conv1d",
            "nn.conv2d",
            "nn.conv2d_transpose",
            "nn.conv3d",
            "nn.conv3d_transpose",
            "nn.dense",
            "nn.batch_matmul",
            "sum",
            "prod",
            "max",
            "min",
            "mean",
        }
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op) in compute_intensive_ops:
                self.is_compute_intensive = True

        return super().visit_call(call)

    def is_graph_compute_intensive(self, subgraph: relay.expr.Expr) -> bool:
        """
        This function recursively visits the graph and checks if it's compute intensive"
        """
        self.visit(subgraph)
        return self.is_compute_intensive


def is_valid_subgraph(params: List[relay.expr.Var], body: relay.expr.Expr) -> bool:
    """Final check on whether the subgraph is valid and should be offloaded to TensorRT."""
    # Remove invalid subgraphs for implicit batch mode.
    if get_tensorrt_use_implicit_batch_mode():
        input_batch_sizes = []
        for var in params:
            # In implicit batch mode, all inputs must have same batch size
            # TODO: (codeislife99) : Fix different dynamic batch size inputs
            if isinstance(var.checked_type, relay.TupleType):
                for tupe_type in var.checked_type.fields:
                    # Scalar inputs not allowed
                    if len(tupe_type.shape) == 0:
                        logger.info("tensorrt: scalar inputs not supported")
                        return False

                    if not isinstance(tupe_type.shape[0], tvm.tir.expr.Any):
                        input_batch_sizes.append(int(tupe_type.shape[0]))
            else:
                # Scalar inputs not allowed
                if len(var.checked_type.shape) == 0:
                    logger.info("tensorrt: scalar inputs not supported")
                    return False
                if not isinstance(var.checked_type.shape[0], tvm.tir.expr.Any):
                    input_batch_sizes.append(int(var.checked_type.shape[0]))

        if len(input_batch_sizes) > 1 and len(set(input_batch_sizes)) != 1:
            logger.info("tensorrt: inputs have different batch sizes: %s", input_batch_sizes)
            return False

    if get_tensorrt_remove_no_mac_subgraphs():
        if not IsComputeIntensiveGraph().is_graph_compute_intensive(body):
            logger.info("tensorrt: not a compute-intensize sub-graph")
            return False

    return True


def prune_tensorrt_subgraphs(mod: tvm.IRModule) -> tvm.IRModule:
    """
    Un-partition those partitions which:
     - have no multiply-accumulates (if remove_no_mac_subgraphs is True)
     - can't actually be supported by TensorRT now that we see the whole partition."""
    global_vars_to_inline = [
        gv
        for gv in mod.get_global_vars()
        if mod[gv].attrs
        and mod[gv].attrs["Compiler"] == "tensorrt"
        and not is_valid_subgraph(mod[gv].params, mod[gv].body)
    ]
    return relay.transform.InlineCompilerFunctionsBoundTo(global_vars_to_inline)(mod)


class RemoveDropout(ExprMutator):
    """
    Removes all nn.dropout from an expr.
    """

    def visit_tuple_getitem(self, op: TupleGetItem) -> relay.expr.Expr:
        visit = super().visit_tuple_getitem(op)
        if visit.index != 0:
            return visit
        if (
            isinstance(visit.tuple_value, Call)
            and isinstance(visit.tuple_value.op, Op)
            and visit.tuple_value.op.name == "nn.dropout"
            and visit.index == 0
        ):
            return visit.tuple_value.args[0]
        return visit


@transform.function_pass(opt_level=0)
class RemoveDropoutPass:
    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, _: tvm.transform.PassContext
    ) -> relay.function.Function:
        return RemoveDropout().visit(func)
