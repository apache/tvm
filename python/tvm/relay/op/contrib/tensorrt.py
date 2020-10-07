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
# pylint: disable=invalid-name, unused-argument
"""TensorRT supported operators."""
import logging
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar
from tvm.relay.expr_functor import ExprMutator

logger = logging.getLogger("TensorRT")


def is_tensorrt_runtime_enabled():
    """Check if the TensorRT graph runtime is present.
    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_tensorrt_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def get_tensorrt_version():
    """Gets the version of TensorRT that TVM is built against or is targeting.

    Returns
    -------
    ret: Tuple[int, int, int]
        TensorRT version as a tuple of major, minor, and patch number. If TVM
        is not built with TensorRT, the value set by set_tensorrt_version() is returned instead.
    """
    pass_ctx = tvm.transform.PassContext.current()
    if "relay.ext.tensorrt.options" in pass_ctx.config:
        return tuple(pass_ctx.config["relay.ext.tensorrt.options"].tensorrt_version)
    return tuple(tvm.get_global_func("relay.op.get_tensorrt_version")())


def get_tensorrt_use_implicit_batch_mode():
    pass_ctx = tvm.transform.PassContext.current()
    if "relay.ext.tensorrt.options" in pass_ctx.config:
        return pass_ctx.config["relay.ext.tensorrt.options"].use_implicit_batch
    logger.warning(
        "PassContext has no relay.ext.tensorrt.options config, using default value "
        "use_implicit_batch=True."
    )
    return True


def get_tensorrt_remove_no_mac_subgraphs():
    pass_ctx = tvm.transform.PassContext.current()
    if "relay.ext.tensorrt.options" in pass_ctx.config:
        return pass_ctx.config["relay.ext.tensorrt.options"].remove_no_mac_subgraphs
    logger.warning(
        "PassContext has no relay.ext.tensorrt.options config, using default value "
        "remove_no_mac_subgraphs=False."
    )
    return False


def partition_for_tensorrt(
    mod,
    params=None,
    version=None,
    use_implicit_batch=True,
    remove_no_mac_subgraphs=False,
    max_workspace_size=1 << 30,
):
    """Partition the graph greedily offloading supported operators to TensorRT.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    version : Optional[Tuple[int, int, int]]
        TensorRT version to target as tuple of (major, minor, patch). If TVM is compiled with
        USE_TENSORRT_RUNTIME=ON, the linked TensorRT version will be used instead.
    use_implicit_batch : Optional[bool]
        Use TensorRT implicit batch mode (default true). Setting to false will enable explicit batch
        mode which will widen supported operators to include those which modify the batch dimension,
        but may reduce performance for some models.
    remove_no_mac_subgraphs : Optional[bool]
        Removes subgraphs which have been partitioned for TensorRT if they do not have any
        multiply-accumulate operations. The removed subgraphs will go through TVM's standard
        compilation instead. Can improve performance.
    max_workspace_size : Optional[int]
        How many bytes of workspace size to allow each subgraph to use for TensorRT engine creation.
        See TensorRT documentation for more info.
    Returns
    -------
    mod_and_config : Tuple[Module, Dict[str, Any]]
        A tuple of 1) annotated and partitioned module and 2) "relay.ext.tensorrt.options"
        configuration which should be given to PassContext when building.
    """
    config = {
        "use_implicit_batch": use_implicit_batch,
        "max_workspace_size": max_workspace_size,
        "remove_no_mac_subgraphs": remove_no_mac_subgraphs,
    }
    if version:
        assert isinstance(version, tuple) and len(version) == 3
        config["tensorrt_version"] = version
    else:
        linked_version = tuple(tvm.get_global_func("relay.op.get_tensorrt_version")())
        if not linked_version:
            logger.warning(
                "TVM was not built against TensorRT and no version was provided to "
                "partition_for_tensorrt. Defaulting to 6.0.1"
            )
            linked_version = (6, 0, 1)
        config["tensorrt_version"] = linked_version

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            RemoveDropoutPass(),
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(
                {"nn.conv2d": ["NCHW", "default"], "nn.conv3d": ["NCDHW", "default"]}
            ),
            transform.FoldConstant(),
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        mod = seq(mod)
        mod = prune_tensorrt_subgraphs(mod)
    return mod, config


def _register_external_op_helper_with_checker(op_name, checker):
    @tvm.ir.register_op_attr(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        if any([x.checked_type.dtype != "float32" for x in args]):
            logger.info("Only float32 inputs are supported for TensorRT.")
            return False
        return checker(attrs, args, op_name)

    return _func_wrapper


def _register_external_op_helper(op_name, supported=True):
    return _register_external_op_helper_with_checker(
        op_name, lambda attrs, args, op_name: supported
    )


# Ops which are always supported
_register_external_op_helper("nn.relu")
_register_external_op_helper("sigmoid")
_register_external_op_helper("tanh")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")
_register_external_op_helper("divide")
_register_external_op_helper("power")
_register_external_op_helper("maximum")
_register_external_op_helper("minimum")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("abs")
_register_external_op_helper("negative")
_register_external_op_helper("nn.batch_flatten")
_register_external_op_helper("clip")


@tvm.ir.register_op_attr("add", "target.tensorrt")
def add_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if add is supported by TensorRT."""

    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if (
        not get_tensorrt_use_implicit_batch_mode()
        and (isinstance(args[0], Constant) or isinstance(args[1], Constant))
        and args[0].checked_type.shape[0] == args[1].checked_type.shape[0]
        and args[0].checked_type.shape[0] != 1
        and (len(args[0].checked_type.shape) > 3 or len(args[1].checked_type.shape) > 3)
    ):
        logger.info("add: bug in TRT with adding batched constants.")
        return False
    return True


@tvm.ir.register_op_attr("nn.batch_norm", "target.tensorrt")
def batch_norm_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.batch_norm is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if int(attrs.axis) not in (1, 3):
        logger.info("nn.batch_norm: axis is %d but must be 1 or 3.", int(attrs.axis))
        return False
    return True


@tvm.ir.register_op_attr("nn.softmax", "target.tensorrt")
def softmax_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.softmax is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axis) == 0:
        logger.info("nn.softmax: can't modify batch dimension.")
        return False
    return True


@tvm.ir.register_op_attr("nn.conv2d", "target.tensorrt")
def conv2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.conv2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.data_layout != "NCHW":
        logger.info("nn.conv2d: data_layout is %s but must be NCHW.", attrs.data_layout)
        return False
    if attrs.kernel_layout != "OIHW":
        logger.info("nn.conv2d: kernel_layout is %s but must be OIHW.", attrs.kernel_layout)
        return False
    if attrs.out_layout and attrs.out_layout != "NCHW":
        logger.info("nn.conv2d: out_layout is %s but must be NCHW.", attrs.out_layout)
        return False
    return True


@tvm.ir.register_op_attr("nn.dense", "target.tensorrt")
def dense_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if dense is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    input_rank = len(args[0].checked_type.shape)
    weight_rank = len(args[1].checked_type.shape)
    if input_rank not in (2, 3, 4):
        logger.info("nn.dense: input has rank %d but must be 2, 3 or 4.", input_rank)
        return False
    if weight_rank != 2:
        logger.info("nn.dense: weight has rank %d but must be 2.", weight_rank)
        return False
    return True


@tvm.ir.register_op_attr("nn.bias_add", "target.tensorrt")
def bias_add_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.bias_add is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    input_rank = len(args[0].checked_type.shape)
    if input_rank not in (2, 3, 4):
        logger.info("nn.bias_add: input rank is %d but must be 2, 3 or 4.", input_rank)
        return False
    return True


@tvm.ir.register_op_attr("nn.max_pool2d", "target.tensorrt")
def max_pool_2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.max_pool2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.layout != "NCHW":
        logger.info("nn.max_pool2d: layout is %s but must be NCHW.", attrs.layout)
        return False
    if attrs.ceil_mode and get_tensorrt_version() < (5, 1, 5):
        logger.info("nn.avg_pool2d: ceil_mode=True requires TensorRT 5.1.5 or greater.")
        return False
    return True


@tvm.ir.register_op_attr("nn.avg_pool2d", "target.tensorrt")
def avg_pool_2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.avg_pool2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.layout != "NCHW":
        logger.info("nn.avg_pool2d: layout is %d but must be NCHW.", attrs.layout)
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
            "nn.avg_pool2d: inclusive-counted blended or average "
            "pooling is not supported in combination with asymmetric padding"
        )
        return False
    if attrs.ceil_mode and get_tensorrt_version() < (5, 1, 5):
        logger.info("nn.avg_pool2d: ceil_mode=True requires TensorRT 5.1.5 or greater.")
        return False
    return True


@tvm.ir.register_op_attr("nn.global_max_pool2d", "target.tensorrt")
def global_max_pool_2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.global_max_pool2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.layout != "NCHW":
        logger.info("nn.global_max_pool2d: layout is %s but must be NCHW.", attrs.layout)
        return False
    return True


@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.tensorrt")
def global_avg_pool_2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.global_avg_pool2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.layout != "NCHW":
        logger.info("nn.global_avg_pool2d: layout is %s but must be NCHW.", attrs.layout)
        return False
    return True


@tvm.ir.register_op_attr("expand_dims", "target.tensorrt")
def expand_dims_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if expand_dims is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axis) == 0:
        logger.info("expand_dims: can't modify batch dimension.")
        return False
    return True


@tvm.ir.register_op_attr("squeeze", "target.tensorrt")
def squeeze_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if squeeze is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not attrs.axis:
        logger.info("squeeze: must explicitly set axis.")
        return False
    if get_tensorrt_use_implicit_batch_mode() and any([axis == 0 for axis in map(int, attrs.axis)]):
        logger.info("squeeze: can't modify batch dimension.")
        return False
    return True


@tvm.ir.register_op_attr("concatenate", "target.tensorrt")
def concatenate_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if concatenate is supported by TensorRT."""
    if any([x.dtype != "float32" for x in args[0].checked_type.fields]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not get_tensorrt_use_implicit_batch_mode():
        return True
    if int(attrs.axis) == 0:
        logger.info("concatenate: can't modify batch dimension.")
        return False
    if isinstance(args[0], Tuple):
        for tuple_input in args[0].fields:
            if isinstance(tuple_input, Constant):
                logger.info("concatenate: can't concatenate tensors with constants.")
                return False
    return True


@tvm.ir.register_op_attr("nn.conv2d_transpose", "target.tensorrt")
def conv2d_transpose_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.conv2d_transpose is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.data_layout != "NCHW":
        logger.info("nn.conv2d_transpose: data_layout is %s but must be NCHW.", attrs.data_layout)
        return False
    if attrs.kernel_layout != "OIHW":
        logger.info(
            "nn.conv2d_transpose: kernel_layout is %s but must be OIHW.", attrs.kernel_layout
        )
        return False
    if attrs.out_layout and attrs.out_layout != "NCHW":
        logger.info("nn.conv2d_transpose: out_layout is %s but must be NCHW.", attrs.out_layout)
        return False
    if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
        logger.info("nn.conv2d_transpose: dilation rate must be 1.")
        return False
    return True


@tvm.ir.register_op_attr("transpose", "target.tensorrt")
def transpose_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if transpose is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if get_tensorrt_use_implicit_batch_mode() and int(attrs.axes[0]) != 0:
        logger.info("transpose: can't modify batch dimension.")
        return False
    return True


@tvm.ir.register_op_attr("layout_transform", "target.tensorrt")
def layout_transform_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if layout_transform is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if (attrs.src_layout, attrs.dst_layout) not in [
        ("NCHW", "NHWC"),
        ("NHWC", "NCHW"),
        ("NDHWC", "NCDHW"),
        ("NCDHW", "NDHWC"),
    ]:
        logger.info(
            "layout_transform: %s to %s is not supported.", attrs.src_layout, attrs.dst_layout
        )
        return False
    return True


@tvm.ir.register_op_attr("reshape", "target.tensorrt")
def reshape_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if reshape is supported by TensorRT."""
    if args[0].checked_type.dtype != "float32":
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if any([x < -1 for x in map(int, attrs.newshape)]):
        logger.info("reshape: new shape dims must be explicit.")
        return False
    if get_tensorrt_use_implicit_batch_mode():
        shape = list(map(int, args[0].checked_type.shape))
        new_shape = list(map(int, attrs.newshape))
        if len(new_shape) == 0 or len(shape) == 0:
            logger.info("reshape: Can't reshape to or from scalar.")
            return False
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
        if shape[0] != new_shape[0]:
            logger.info("reshape: can't modify batch dimension.")
            return False
    return True


@tvm.ir.register_op_attr("nn.pad", "target.tensorrt")
def pad_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.pad is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if attrs.pad_mode != "constant":
        logger.info("nn.pad: pad mode is %s but must be constant.", attrs.pad_mode)
        return False
    if float(attrs.pad_value) != 0.0:
        logger.info("nn.pad: pad value is %f but must be 0.0.", float(attrs.pad_value))
        return False
    if any([x != 0 for x in attrs.pad_width[0]]) or any([x != 0 for x in attrs.pad_width[1]]):
        logger.info("nn.pad: can't pad batch or channel dimensions.")
        return False
    if len(attrs.pad_width) == 5 and any([x != 0 for x in attrs.pad_width[2]]):
        logger.info("nn.pad: can only pad last two dimensions for 5D inputs.")
    return True


def reduce_annotate_fn(attrs, args, op_name):
    """Helper for reduce operations."""
    if not attrs.axis or len(attrs.axis) == 0:
        logger.info("%s: cannot reduce to scalar.", op_name)
        return False
    if attrs.exclude:
        logger.info("%s: exclude not supported.", op_name)
        return False
    if get_tensorrt_use_implicit_batch_mode() and any([x == 0 for x in map(int, attrs.axis)]):
        logger.info("%s: can't modify batch dimension.", op_name)
        return False
    return True


_register_external_op_helper_with_checker("sum", reduce_annotate_fn)
_register_external_op_helper_with_checker("prod", reduce_annotate_fn)
_register_external_op_helper_with_checker("max", reduce_annotate_fn)
_register_external_op_helper_with_checker("min", reduce_annotate_fn)
_register_external_op_helper_with_checker("mean", reduce_annotate_fn)


def trt_version_annotate_fn(version):
    """Helper for ops which require a minimum TRT version"""

    def _func_wrapper(attrs, args, op_name):
        if get_tensorrt_version() < version:
            logger.info(
                "%s: requires TensorRT version %s or higher.", op_name, ".".join(map(str, version))
            )
            return False
        return True

    return _func_wrapper


_register_external_op_helper_with_checker("nn.leaky_relu", trt_version_annotate_fn((5, 1, 5)))
_register_external_op_helper_with_checker("sin", trt_version_annotate_fn((5, 1, 5)))
_register_external_op_helper_with_checker("cos", trt_version_annotate_fn((5, 1, 5)))
_register_external_op_helper_with_checker("atan", trt_version_annotate_fn((5, 1, 5)))
_register_external_op_helper_with_checker("ceil", trt_version_annotate_fn((5, 1, 5)))


@tvm.ir.register_op_attr("strided_slice", "target.tensorrt")
def strided_slice_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if strided_slice is supported by TensorRT."""
    if args[0].checked_type.dtype != "float32":
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not trt_version_annotate_fn((5, 1, 5))(attrs, args, "strided_slice"):
        return False
    if get_tensorrt_use_implicit_batch_mode():
        batch_dim_begin_modified = attrs.begin[0] is not None and int(attrs.begin[0]) != 0
        batch_dim_end_modified = (
            attrs.end[0] is not None
            and int(attrs.end[0]) != -1
            and int(attrs.end[0]) != int(args[0].checked_type.shape[0])
        )
        if batch_dim_begin_modified or batch_dim_end_modified:
            logger.info("strided_slice: can't modify batch dimension.")
            return False
    if any([x is not None and x <= 0 for x in attrs.strides]):
        logger.info("strided_slice: stride must be positive")
        return False
    return True


@tvm.ir.register_op_attr("nn.adaptive_max_pool2d", "target.tensorrt")
def adapative_max_pool2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.adaptive_max_pool2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if len(attrs.output_size) == 0 or any([size != 1 for size in map(int, attrs.output_size)]):
        logger.info("nn.adaptive_max_pool2d: output size must be (1, 1).")
        return False
    return True


@tvm.ir.register_op_attr("nn.adaptive_avg_pool2d", "target.tensorrt")
def adapative_avg_pool2d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.adaptive_avg_pool2d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if len(attrs.output_size) == 0 or any([size != 1 for size in map(int, attrs.output_size)]):
        logger.info("nn.adaptive_avg_pool2d: output size must be (1, 1).")
        return False
    return True


@tvm.ir.register_op_attr("nn.conv3d", "target.tensorrt")
def conv3d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.conv3d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not trt_version_annotate_fn((6, 0, 1))(attrs, args, "nn.conv3d"):
        return False
    if attrs.data_layout != "NCDHW":
        logger.info("nn.conv3d: data_layout is %s but must be NCDHW.", attrs.data_layout)
        return False
    if attrs.kernel_layout != "OIDHW":
        logger.info("nn.conv3d: kernel_layout is %s but must be OIDHW.", attrs.kernel_layout)
        return False
    if attrs.out_layout and attrs.out_layout != "NCDHW":
        logger.info("nn.conv3d: out_layout is %s but must be NCDHW.", attrs.out_layout)
        return False
    return True


@tvm.ir.register_op_attr("nn.max_pool3d", "target.tensorrt")
def max_pool_3d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.max_pool3d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not trt_version_annotate_fn((6, 0, 1))(attrs, args, "nn.max_pool3d"):
        return False
    if attrs.layout != "NCDHW":
        logger.info("nn.max_pool3d: layout is %s but must be NCDHW.", attrs.layout)
        return False
    return True


@tvm.ir.register_op_attr("nn.avg_pool3d", "target.tensorrt")
def avg_pool_3d_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.avg_pool3d is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not trt_version_annotate_fn((6, 0, 1))(attrs, args, "nn.avg_pool3d"):
        return False
    if attrs.layout != "NCDHW":
        logger.info("nn.avg_pool3d: layout is %s but must be NCDHW.", attrs.layout)
        return False
    return True


@tvm.ir.register_op_attr("nn.conv3d_transpose", "target.tensorrt")
def conv3d_transpose_annotate_fn(attrs, args):  # pylint: disable=unused-variable
    """Check if nn.conv3d_transpose is supported by TensorRT."""
    if any([x.checked_type.dtype != "float32" for x in args]):
        logger.info("Only float32 inputs are supported for TensorRT.")
        return False
    if not trt_version_annotate_fn((6, 0, 1))(attrs, args, "nn.conv3d_transpose"):
        return False
    if attrs.data_layout != "NCDHW":
        logger.info("nn.conv3d_transpose: data_layout is %s but must be NCDHW.", attrs.data_layout)
        return False
    if attrs.kernel_layout != "OIDHW":
        logger.info(
            "nn.conv3d_transpose: kernel_layout is %s but must be OIDHW.", attrs.kernel_layout
        )
        return False
    if attrs.out_layout and attrs.out_layout != "NCDHW":
        logger.info("nn.conv3d_transpose: out_layout is %s but must be NCDHW.", attrs.out_layout)
        return False
    if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
        logger.info("nn.conv3d_transpose: dilation rate must be 1.")
        return False
    if attrs.output_padding and any([x != 0 for x in map(int, attrs.output_padding)]):
        logger.info("nn.conv3d_transpose: output padding is not supported.")
        return False
    return True


def is_valid_subgraph(params, body):
    """Final check on whether the subgraph is valid and should be offloaded to TensorRT."""
    # Remove invalid subgraphs for implicit batch mode.
    if get_tensorrt_use_implicit_batch_mode():
        input_batch_sizes = []
        for var in params:
            # In implicit batch mode, all inputs must have same batch size
            if isinstance(var.checked_type, relay.TupleType):
                for tupe_type in var.checked_type.fields:
                    # Scalar inputs not allowed
                    if len(tupe_type.shape) == 0:
                        logger.info("tensorrt: scalar inputs not supported")
                        return False
                    input_batch_sizes.append(int(tupe_type.shape[0]))
            else:
                # Scalar inputs not allowed
                if len(var.checked_type.shape) == 0:
                    logger.info("tensorrt: scalar inputs not supported")
                    return False
                input_batch_sizes.append(int(var.checked_type.shape[0]))
        if len(input_batch_sizes) > 1 and len(set(input_batch_sizes)) != 1:
            logger.info("tensorrt: inputs have different batch sizes")
            return False
    # Remove subgraphs with no multiply-accumulates
    if get_tensorrt_remove_no_mac_subgraphs() and relay.analysis.get_total_mac_number(body) == 0:
        return False
    return True


def prune_tensorrt_subgraphs(mod):
    """
    Removes invalid subgraphs and those with no multiply-accumulates (if remove_no_max_subgraphs
    is set).
    """

    class SubgraphRemover(ExprMutator):
        """
        Reverts subgraphs in subgraphs_to_remove back to TVM instead of using an external codegen.
        """

        def __init__(self, subgraphs_to_remove, mod, new_mod):
            ExprMutator.__init__(self)
            self.subgraphs_to_remove = subgraphs_to_remove
            self.mod = mod
            self.new_mod = new_mod

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar):
                name = call.op.name_hint
                if name in self.subgraphs_to_remove:
                    # "Inline" the subgraph back into new main function.
                    func = self.mod[name]
                    var_map = {}
                    for arg, param in zip(call.args, func.params):
                        var_map[param] = super().visit(arg)
                    new_body = relay.bind(func.body, var_map)
                    return new_body
                if name != "main":
                    # Copy the GlobalVar (subgraph function) to the new module and call.
                    args = []
                    for arg in call.args:
                        args.append(super().visit(arg))
                    subgraph_gv = relay.GlobalVar(name)
                    self.new_mod[subgraph_gv] = self.mod[name]
                    return subgraph_gv(*args)
            return super().visit_call(call)

    subgraphs_to_remove = []
    # Remove invalid subgraphs
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != "tensorrt":
            continue
        if not is_valid_subgraph(mod[name].params, mod[name].body):
            subgraphs_to_remove.append(name)
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraphs_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod


class RemoveDropout(ExprMutator):
    """
    Removes all nn.dropout from an expr.
    """

    def visit_tuple_getitem(self, op):
        visit = super().visit_tuple_getitem(op)
        if (
            isinstance(visit.tuple_value, Call)
            and visit.tuple_value.op.name == "nn.dropout"
            and visit.index == 0
        ):
            return visit.tuple_value.args[0]
        return visit


@transform.function_pass(opt_level=0)
class RemoveDropoutPass:
    def transform_function(self, func, mod, _):
        return RemoveDropout().visit(func)
