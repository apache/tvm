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
# pylint: disable=invalid-name, unused-argument, pointless-exception-statement
"""CLML Library supported operators."""
import json
from string import Template
import numpy as np
import tvm

from tvm import relay
from tvm.ir import Op
from tvm._ffi import register_func
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import function as _function
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import Call, TupleGetItem, Var, Constant

from ...dataflow_pattern import wildcard, is_op, is_constant, is_tuple_get_item, is_tuple
from .register import register_pattern_table
from ..strategy.generic import is_depthwise_conv2d


def clml_sdk_version():
    """Utility function to get clml version"""

    return int(tvm.support.libinfo().get("TVM_CLML_VERSION", 2))


def is_clml_runtime_enabled():
    """Check if the CLML graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_clml_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


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


class OptimizeBatchnorm(ExprMutator):
    """
    Fuse Conv+Batchnorm and constant folder to generate Conv+Add.
    """

    def visit_call(self, call) -> relay.expr.Expr:
        new_args = []
        for arg in call.args:
            if (
                not isinstance(arg, (Var, Constant))
                and isinstance(arg, tvm.relay.TupleGetItem)
                and arg.tuple_value.op.name == "nn.batch_norm"
                and (not isinstance(arg.tuple_value.args[0], (Var, Constant)))
                and arg.tuple_value.args[0].op.name == "nn.conv2d"
            ):
                ep = arg.tuple_value.attrs["epsilon"]
                wt = arg.tuple_value.args[1].data.numpy()
                bs = arg.tuple_value.args[2].data.numpy()
                mn = arg.tuple_value.args[3].data.numpy()
                vr = arg.tuple_value.args[4].data.numpy() + ep
                dino = np.sqrt(vr)
                wt = wt / dino
                bs = bs - mn * wt
                conv_op = arg.tuple_value.args[0]
                conv_args = list(conv_op.args)
                wt_conv = conv_args[1].data.numpy()
                if conv_op.attrs["kernel_layout"] == "OIHW":
                    wt = wt.reshape(wt.shape[0], 1, 1, 1)
                elif conv_op.attrs["kernel_layout"] == "IOHW":
                    wt = wt.reshape(1, wt.shape[0], 1, 1)
                else:
                    raise ValueError("Unsupported Conv2d kernel layout")
                wt_conv = wt_conv * wt
                conv_args[1] = relay.const(tvm.nd.array(wt_conv))
                bs_args = relay.const(tvm.nd.array(bs.reshape(-1, bs.shape[0], 1, 1)))
                conv_out = Call(
                    arg.tuple_value.args[0].op, conv_args, arg.tuple_value.args[0].attrs
                )
                mod = tvm.relay.add(conv_out, bs_args)
                new_args.append(mod)
            else:
                new_args.append(arg)

        call = Call(call.op, new_args, call.attrs)
        args = [self.visit(arg) for arg in call.args]

        return Call(call.op, args, call.attrs)


@transform.function_pass(opt_level=0)
class OptimizeBatchnormPass:
    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, _: tvm.transform.PassContext
    ) -> relay.function.Function:
        return OptimizeBatchnorm().visit(func)


def partition_for_clml(mod, params=None, **opts):
    """Partition the graph greedily offloading supported
    operators to CLML Library.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            RemoveDropoutPass(),
            transform.FoldConstant(),
            OptimizeBatchnormPass(),
            transform.MergeComposite(clml_pattern_table()),
            transform.AnnotateTarget("clml", False),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    result_mod = seq(mod)
    return result_mod


@register_func("relay.ext.clml.optimize")
def preprocess_module(mod):
    """
    Pre-process a module containing functions ready for CLML codegen. For now we enforce OIHW
    kernel layout and fold the transforms away.

    Parameters
    ----------
    mod : Module
        The module to run passes on.

    Returns
    -------
    preprocessed_mod : The processed module.
    """

    def alter_conv(attrs, inputs, tinfos, out_type):
        new_attrs = dict(attrs)
        data_info = tinfos[0]
        weight_info = tinfos[1]
        (desired_data_layout, desired_kernel_layout) = ("NCHW", "OIHW")
        new_attrs["data_layout"] = desired_data_layout
        new_attrs["kernel_layout"] = desired_kernel_layout

        if is_depthwise_conv2d(
            data_info.shape,
            attrs["data_layout"],
            weight_info.shape,
            attrs["kernel_layout"],
            attrs["groups"],
        ):
            dkl = desired_kernel_layout
            new_attrs["kernel_layout"] = dkl[1] + dkl[0] + dkl[2] + dkl[3]
        return relay.nn.conv2d(*inputs, **new_attrs)

    with OpAttrContext("nn.conv2d", "FTVMAlterOpLayout", alter_conv):
        seq = tvm.transform.Sequential(
            [
                transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"]}),
                transform.ConvertLayout({"nn.conv2d_transpose": ["NCHW", "OIHW"]}),
                transform.AlterOpLayout(),
                transform.FoldConstant(),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            preprocessed_mod = seq(mod)
    return preprocessed_mod


def preprocess_for_clml(mod):
    """Preprocessing pass to alter the layouts for CLML compiler target"""

    for _var in mod.get_global_vars():
        if _var.name_hint == "main":
            continue
        fn = mod[_var.name_hint]
        if "Compiler" in fn.attrs.keys() and fn.attrs["Compiler"] == "clml":
            new_fn = fn.body
            clml_mod = tvm.IRModule.from_expr(new_fn)
            with tvm.transform.PassContext(opt_level=3):
                clml_mod = preprocess_module(clml_mod)
            new_body = clml_mod["main"].body
            mod[_var.name_hint] = _function.Function(
                fn.params, new_body, fn.ret_type, fn.type_params, fn.attrs
            )
    return mod


@register_pattern_table("clml")
def clml_pattern_table():
    """Get the CLML pattern table."""

    def conv_pattern():
        """Create a convolution pattern."""
        pattern = is_op("nn.conv2d")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(
            lambda x: is_tuple_get_item(
                is_op("nn.batch_norm")(
                    x, is_constant(), is_constant(), is_constant(), is_constant()
                )
            )
        )
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def conv_transpose_pattern():
        """Create a transposed convolution pattern."""
        pattern = is_op("nn.conv2d_transpose")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(
            lambda x: is_tuple_get_item(
                is_op("nn.batch_norm")(
                    x, is_constant(), is_constant(), is_constant(), is_constant()
                )
            )
        )
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def pad_conv_pattern():
        """Create a pad with convolution pattern."""
        pattern = is_op("nn.pad")(wildcard(), is_constant())
        pattern = is_op("nn.conv2d")(pattern, is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(
            lambda x: is_tuple_get_item(
                is_op("nn.batch_norm")(
                    x, is_constant(), is_constant(), is_constant(), is_constant()
                )
            )
        )
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def batch_norm_pattern():
        """Create a batch norm pattern."""
        pattern = is_op("nn.batch_norm")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_tuple_get_item(pattern)
        return pattern

    def concat_pattern():
        """Create a concat pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the concat pattern.
        """
        pattern = is_tuple(None)
        pattern = is_op("concatenate")(pattern)

        return pattern

    def dense1d_pattern():
        """Create a dense pattern for 1d vector to matrix multiple."""
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        return pattern

    def dense2d_pattern():
        """Create a dense pattern for 2d matrix to matrix multiple."""
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        return pattern

    def pad_pattern():
        """Create a pad pattern."""
        pattern = is_op("nn.pad")(wildcard(), is_constant())
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by CLML."""
        call = extract
        clip_found = False
        if isinstance(call, tvm.relay.expr.TupleGetItem):
            call = call.tuple_value
        elif call.op.name == "nn.relu":
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value
        elif call.op.name == "clip":
            clip_found = True
            if call.attrs["a_min"] != 0.0 or call.attrs["a_max"] != 6.0:
                return False
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value

        while call.op.name != "nn.conv2d":
            call = call.args[0]

        attrs, args = call.attrs, call.args
        if attrs.data_layout != "NCHW":
            return False

        if (
            (not clip_found)
            and (attrs.kernel_size[0] == 3)
            and (attrs.dilation[0] != 1)
            and (attrs.groups != 1)
            and (attrs.channels == attrs.groups)
        ):
            return False

        data_typ = args[0].checked_type
        kernel_typ = args[1].checked_type
        is_depthwise = is_depthwise_conv2d(
            data_typ.shape,
            attrs["data_layout"],
            kernel_typ.shape,
            attrs["kernel_layout"],
            attrs["groups"],
        )
        if attrs.groups != 1 and not is_depthwise:
            return False
        return True

    def check_conv_transpose(extract):
        """Check transposed conv pattern is supported by CLML."""
        call = extract
        if isinstance(call, tvm.relay.expr.TupleGetItem):
            call = call.tuple_value
        elif call.op.name == "nn.relu":
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value
        elif call.op.name == "clip":
            if call.attrs["a_min"] != 0.0 or call.attrs["a_max"] != 6.0:
                return False
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value

        while call.op.name != "nn.conv2d_transpose":
            call = call.args[0]

        attrs = call.attrs
        if attrs.data_layout != "NCHW":
            return False

        return True

    def check_binary_op(extract):
        call = extract
        # Scalars are not supported
        if len(call.args[1].checked_type.shape) == 0:
            return False

        if tuple(call.args[0].checked_type.shape) != tuple(call.args[1].checked_type.shape):
            return False

        for arg in call.args:
            # Avoid any operators with dtype Int64
            if arg.checked_type.dtype == "int64":
                return False
            # No support for batch> 1
            if arg.checked_type.shape[0] > 1:
                return False

        return True

    def check_pad_op(extract):
        call = extract
        if len(call.attrs["pad_width"]) != 4:
            return False
        # CLML can't process Tensor padding with out knowing layout.
        # Pad layers before any convolution are not guarenteed to be NCHW.
        if isinstance(call.args[0], tvm.relay.expr.Var):
            return False
        return True

    def check_softmax_op(extract):
        call = extract
        # supports 2D and 4D tensors
        if len(call.args[0].checked_type.shape) not in [2, 4]:
            return False
        return True

    def check_upsampling_op(extract):
        call = extract
        if call.attrs["method"] != "bilinear":
            return False
        return True

    def check_concat_op(extract):
        call = extract
        if call.attrs["axis"] != 1:
            return False
        return True

    def check_default_op(extract):
        call = extract

        if isinstance(call, tvm.relay.expr.TupleGetItem):
            call = call.tuple_value

        # Avoid any operators with dtype Int64
        for arg in call.args:
            if arg.checked_type.dtype == "int64":
                return False
        return True

    def check_batch_matmul_op(extract):
        call = extract
        # Only support single Matmul
        if call.args[0].checked_type.shape[0] > 1:
            return False
        if call.args[1].checked_type.shape[0] > 1:
            return False
        return True

    def check_dense1d_op(extract):
        call = extract
        # Only support single Matmul
        if call.args[0].checked_type.shape[0] > 1:
            return False
        if not (call.op.name in ["nn.bias_add", "add"] and call.args[0].op.name == "nn.dense"):
            return False
        return True

    def check_reshape(extract):
        call = extract
        call_shape = call.checked_type.shape
        # Only support batch dim = 1
        if call_shape[0] > 1:
            return False
        # Checking buffer indexing limit
        for shape in call_shape:
            if shape > 32768:
                return False
        return True

    return [
        ("clml.pad_conv2d", pad_conv_pattern(), check_conv),
        ("clml.conv2d", conv_pattern(), check_conv),
        ("clml.conv2d_transpose", conv_transpose_pattern(), check_conv_transpose),
        ("clml.dense1d", dense1d_pattern(), check_dense1d_op),
        ("clml.dense2d", dense2d_pattern(), check_default_op),
        ("clml.pad", pad_pattern(), check_pad_op),
        ("clml.concat", concat_pattern(), check_concat_op),
        ("clml.batch_norm", batch_norm_pattern(), check_default_op),
        ("clml.add", is_op("add")(wildcard(), wildcard()), check_binary_op),
        ("clml.subtract", is_op("subtract")(wildcard(), wildcard()), check_binary_op),
        ("clml.multiply", is_op("multiply")(wildcard(), wildcard()), check_binary_op),
        ("clml.divide", is_op("divide")(wildcard(), wildcard()), check_binary_op),
        ("clml.minimum", is_op("minimum")(wildcard(), wildcard()), check_binary_op),
        ("clml.maximum", is_op("maximum")(wildcard(), wildcard()), check_binary_op),
        ("clml.softmax", is_op("nn.softmax")(wildcard()), check_softmax_op),
        ("clml.reshape", is_op("reshape")(wildcard()), check_reshape),
        ("clml.avg_pool2d", is_op("nn.avg_pool2d")(wildcard()), check_default_op),
        ("clml.max_pool2d", is_op("nn.max_pool2d")(wildcard()), check_default_op),
        ("clml.global_avg_pool2d", is_op("nn.global_avg_pool2d")(wildcard()), check_default_op),
        ("clml.global_max_pool2d", is_op("nn.global_max_pool2d")(wildcard()), check_default_op),
        ("clml.relu", is_op("nn.relu")(wildcard()), check_default_op),
        ("clml.clip", is_op("clip")(wildcard()), check_default_op),
        ("clml.batch_flatten", is_op("nn.batch_flatten")(wildcard()), check_default_op),
        ("clml.depth_to_space", is_op("nn.depth_to_space")(wildcard()), check_default_op),
        ("clml.upsampling", is_op("nn.upsampling")(wildcard()), check_upsampling_op),
        (
            "clml.batch_matmul",
            is_op("nn.batch_matmul")(wildcard(), wildcard()),
            check_batch_matmul_op,
        ),
    ]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.clml")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("minimum")
_register_external_op_helper("maximum")


class OpAttrContext(object):
    """Temporarily changes the attr of an op."""

    def __init__(self, op_name, attr_key, attr_value):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        op_name : str
            The op name.

        attr_key : str
            The attribute name.

        attr_value : object
            The attribute value.
        """
        self.op = relay.op.get(op_name)
        self.attr_key = attr_key
        self.attr_value = attr_value

    def __enter__(self):
        self.older_attr = self.op.get_attr(self.attr_key)
        self.op.reset_attr(self.attr_key)
        self.op.set_attr(self.attr_key, self.attr_value)
        return self

    def __exit__(self, ptype, value, trace):
        self.op.reset_attr(self.attr_key)
        if self.older_attr:
            self.op.set_attr(self.attr_key, self.older_attr)


class CLMLGetSubModuleSrc:
    """Generates CLML API one CLML sub module out ot global TVM module"""

    def __init__(self, cmod):
        """Initialize
        Parameters
        ----------
        cmod : Module
            The CLML sub module from TVM module
        """
        self.cmod = cmod
        self.codegen = None
        self.nodes = None
        self.node_map = {}
        self.input_meta = []
        self.output_meta = []
        self.clml_code = []
        self.sub_module_name = None

        self.MakeCLMLTensor = Template(
            """auto $name = runner.MakeCLMLTensor
        (std::vector<size_t>({$shape}), "$dtype", $layout);"""
        )
        self.MapInsert = Template("""runner.storage_map.insert({"$nid", $tensor_desc});""")
        self.MakeConv2D = Template(
            """
        // Convolution / Depthwise Convolution
        runner.MakeConv2D($input_tensor,
           $weight_tensor,
           $bias_tensor,
           $output_tensor,
           std::vector<cl_uint>({$padding}),
           std::vector<cl_uint>({$dilation}),
           std::vector<cl_uint>({$strides}),
           $groups,
           $mode,
           $activation,
           $has_bias,
           $has_act,
           "$dtype");"""
        )
        self.MakeConv2DWithBN = Template(
            """
        // Batchnorm
        runner.MakeConv2DWithBN($input_tensor,
                 $weight_tensor,
                 $bias_tensor,
                 $output_tensor,
                 $bn_scale_tensor,
                 $bn_bias_tensor,
                 $bn_mean_tensor,
                 $bn_var_tensor,
                 std::vector<float>  ({$bn_attrs}),
                 std::vector<cl_uint> ({$padding}),
                 std::vector<cl_uint> ({$dilation}),
                 std::vector<cl_uint> ({$strides}),
                 $groups,
                 $mode,
                 $activation,
                 $has_bias,
                 $has_act,
                 "$dtype");"""
        )
        self.MakeRelu = Template(
            """
        // Relu / Relu6
        runner.MakeRelu($input_tensor, $output_tensor, $relu_type, "$dtype");
        """
        )
        self.MakeBN = Template(
            """
        // Batchnorm
        runner.MakeBatchNorm($input_tensor,
              $output_tensor,
              $bn_scale_tensor,
              $bn_bias_tensor,
              $bn_mean_tensor,
              $bn_var_tensor,
              std::vector<float> ({$bn_attrs}), "$dtype");"""
        )
        self.MakePool2D = Template(
            """
        // Pool2D
        runner.MakePool2D($input_tensor,
           $output_tensor,
           std::vector<cl_uint> ({$pool_size}),
           std::vector<cl_uint> ({$strides}),
           std::vector<cl_uint> ({$padding}),
           "$pool_type", "$dtype");"""
        )
        self.MakeGlobalPool2D = Template(
            """
        // GlobalPool2D
        runner.MakeGlobalPool2D($input_tensor,
                 $output_tensor,
                 std::vector<cl_uint> ({$in_shape}),
                 "$pool_type", "$dtype");"""
        )
        self.MakeReshape = Template(
            """
        // Reshape
        runner.MakeReshape($input_tensor,
            $output_tensor, "$dtype");"""
        )
        self.MakeConcatenate = Template(
            """
        // Concatinate
        runner.MakeConcatenate(
                std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> ({$in_list}),
                $output_tensor,
                $axis, "$dtype");"""
        )
        self.MakeDense = Template(
            """
        // Dense
        runner.MakeDense($input_tensor,
          $weight_tensor,
          $output_tensor,
          std::vector<cl_uint> ({$in_shape}),
          std::vector<cl_uint> ({$wt_shape}),
          "$dtype");"""
        )
        self.MakeSoftMax = Template(
            """
        // Softmax
        runner.MakeSoftMax($input_tensor,
            $output_tensor, "$dtype");"""
        )
        self.MakePad = Template(
            """
        // Pad
        runner.MakePad($input_tensor,
        $output_tensor,
        "$pad_mode",
        std::vector<cl_uint> ({$padding}), "$dtype");"""
        )
        self.MakeBatchFlatten = Template(
            """
        // BatchFlatten
        runner.MakeBatchFlatten($input_tensor,
                 $output_tensor, "$dtype");"""
        )
        self.MakeClip = Template(
            """
        // Clip
        runner.MakeClip($input_tensor,
         $output_tensor,
         $a_max,
         $a_min,
         "$dtype");"""
        )
        self.MakeBinaryOp = Template(
            """
        // BinaryOp
        runner.MakeBinaryOp($input_a,
             $input_b,
             $output_tensor,
             "$op", "$dtype");"""
        )

        self.MakeHeader = Template(
            """
        CLMLRunner $module(std::string name,
                   ToolArgs& args,
                   cl_platform_id arg_platform_id,
                   cl_context arg_context,
                   cl_device_id arg_device_id,
                   cl_command_queue arg_queue) {
        CLMLRunner runner = CLMLRunner(name,
                                 args,
                                 arg_platform_id,
                                 arg_context,
                                 arg_device_id,
                                 arg_queue);
        runner.MakeUnusedTensor();
        """
        )

        self.MakeFooter = Template(
            """
            return runner;
        }
        """
        )

        self.MakeMetaInfo = Template(
            "runner.SetMetaInfo("
            '"Subgraph Name: $name\\n    Input Count  : $input_count\\n'
            "    Output Count : $output_count\\n"
            '    Input MetaInfo\\n$input_meta\\n    Output MetaInfo\\n$output_meta");'
        )
        self.MakeInputMetaInfo = Template(
            "        Input: $in_name\\n          Dtype : $dtype\\n          Shape : [$shape]\\n"
        )

        self.MakeOutputMetaInfo = Template(
            "        Output: $out_name\\n         Dtype : $dtype\\n          Shape : [$shape]\\n"
        )

    def get_src(self):
        """Returns pair of sub module name and the generated source"""

        self.codegen = json.loads(self.cmod.get_source("json"))
        self.sub_module_name = self.codegen["symbol"]
        self.nodes = self.codegen["nodes"]
        self.clml_code.append(self.MakeHeader.substitute(module=self.sub_module_name))

        def get_tensor_from_map(
            node_seq, shape=None, layout="CL_TENSOR_LAYOUT_OPTIMAL_QCOM", dtype="float32"
        ):
            if node_seq in self.node_map:
                return self.node_map[node_seq]
            else:
                node = self.nodes[node_seq]
                dtype = str(node["attrs"]["dtype"][0][0])
                if node["op"] == "input":
                    self.clml_code.append("// Input Node")
                    node_out_name = self.sub_module_name + "_" + "input_" + str(node_seq)
                else:
                    node_out_name = node["name"]
                if shape is None:
                    shape = str(tuple(node["attrs"]["shape"][0][0]))[1:-1]

                self.clml_code.append(
                    self.MakeCLMLTensor.substitute(
                        name=node_out_name, shape=shape, dtype=dtype, layout=layout
                    )
                )
                self.clml_code.append(
                    self.MapInsert.substitute(nid=node_out_name, tensor_desc=node_out_name)
                )
                if node["op"] == "input":
                    self.clml_code.append(
                        Template("runner.inputs.push_back($clml_input);").substitute(
                            clml_input=node_out_name
                        )
                    )
                    self.input_meta.append(
                        self.MakeInputMetaInfo.substitute(
                            in_name=node_out_name, dtype=dtype, shape=shape
                        )
                    )

                if self.nodes[node_seq]["op"] == "const":
                    self.clml_code.append(
                        Template('runner.consts.push_back("$nid");').substitute(nid=node["name"])
                    )
                self.node_map[node_seq] = node_out_name
                return node_out_name

        def make_output_tensor(
            node, node_seq, shape=None, layout="CL_TENSOR_LAYOUT_OPTIMAL_QCOM", dtype="float32"
        ):
            if dtype is None:
                dtype = str(node["attrs"]["dtype"][0][0])
            if shape is None:
                shape = str(tuple(node["attrs"]["shape"][0][0]))[1:-1]
            node_out_name = self.sub_module_name + "_" + "layer_out_" + str(node_seq)
            self.clml_code.append(
                self.MakeCLMLTensor.substitute(
                    name=node_out_name,
                    shape=shape,
                    dtype=dtype,
                    layout=layout,
                )
            )
            return node_out_name

        for node_seq, node in enumerate(self.nodes):
            if node["op"] == "kernel":
                self.clml_code.append("// Kernel Node : " + node["name"])
                if node["name"] == "nn.conv2d" or node["name"] == "nn.depthwise_conv2d":
                    if "padding" in node["attrs"]:
                        padding = str(tuple(int(x) for x in node["attrs"]["padding"][0]))[1:-1]
                    else:
                        padding = "0, 0, 0, 0"
                    dilation = str(tuple(int(x) for x in node["attrs"]["dilation"][0]))[1:-1]
                    strides = str(tuple(int(x) for x in node["attrs"]["strides"][0]))[1:-1]
                    groups = node["attrs"]["groups"][0][0]
                    if node["name"] == "nn.conv2d":
                        mode = "CL_CONVOLUTION_MODE_CONVOLUTION_QCOM"
                    else:
                        mode = "CL_CONVOLUTION_MODE_DEPTHWISE_QCOM"
                    activation = "CL_ACTIVATION_RELU"
                    has_act = False
                    if "activation_type" in node["attrs"]:
                        has_act = True
                        activation = node["attrs"]["activation_type"][0][0]
                        if activation == "relu":
                            activation = "CL_ACTIVATION_RELU"
                        elif activation == "relu6":
                            activation = "CL_ACTIVATION_RELU6"
                        else:
                            raise RuntimeError("Unknown activation:" + activation)
                    has_bias = bool((node["inputs"] == 3) or (node["inputs"] == 7))
                    has_bn = bool((node["inputs"] == 6) or (node["inputs"] == 7))
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    weight_tensor = get_tensor_from_map(node["inputs"][1][0])
                    if not has_bias:
                        bias_tensor = "runner.unusedTensor"
                    else:
                        bias_tensor = get_tensor_from_map(node["inputs"][2][0])

                    node_out_name = make_output_tensor(node, node_seq)

                    if not has_bn:
                        self.clml_code.append(
                            self.MakeConv2D.substitute(
                                input_tensor=input_tensor,
                                weight_tensor=weight_tensor,
                                bias_tensor=bias_tensor,
                                output_tensor=node_out_name,
                                padding=padding,
                                dilation=dilation,
                                strides=strides,
                                groups=groups,
                                mode=mode,
                                activation=activation,
                                has_bias="true" if has_bias else "false",
                                has_act="true" if has_act else "false",
                                dtype=node["attrs"]["dtype"][0][0],
                            )
                        )
                    else:
                        bn_index = 3 if has_bias else 2
                        bn_attrs = tuple(node["attrs"]["batchnorm"][0][0])
                        axis = bn_attrs[0]
                        bn_shape = [1, 1, 1, 1]
                        bn_node = self.nodes[node["inputs"][bn_index][0]]
                        bn_shape[axis] = bn_node["attrs"]["shape"][0][0]
                        dtype = bn_node["attrs"]["dtype"][0][0]

                        bn_scale_tensor = get_tensor_from_map(
                            node["inputs"][bn_index][0],
                            shape=str(tuple(bn_shape))[1:-1],
                            dtype=dtype,
                        )

                        bn_bias_tensor = get_tensor_from_map(
                            node["inputs"][bn_index + 1][0],
                            shape=str(tuple(bn_shape))[1:-1],
                            dtype=dtype,
                        )

                        bn_mean_tensor = get_tensor_from_map(
                            node["inputs"][bn_index + 2][0],
                            shape=str(tuple(bn_shape))[1:-1],
                            dtype=dtype,
                        )

                        bn_var_tensor = get_tensor_from_map(
                            node["inputs"][bn_index + 3][0],
                            shape=str(tuple(bn_shape))[1:-1],
                            dtype=dtype,
                        )

                        self.clml_code.append(
                            self.MakeConv2DWithBN.substitute(
                                input_tensor=input_tensor,
                                weight_tensor=weight_tensor,
                                bias_tensor=bias_tensor,
                                output_tensor=node_out_name,
                                bn_scale_tensor=bn_scale_tensor,
                                bn_bias_tensor=bn_bias_tensor,
                                bn_mean_tensor=bn_mean_tensor,
                                bn_var_tensor=bn_var_tensor,
                                bn_attrs=str(bn_attrs)[1:-1],
                                padding=padding,
                                dilation=dilation,
                                strides=strides,
                                groups=groups,
                                mode=mode,
                                activation=activation,
                                has_bias="true" if has_bias else "false",
                                has_act="true" if has_act else "false",
                                dtype=node["attrs"]["dtype"][0][0],
                            )
                        )
                elif node["name"] == "nn.relu6" or node["name"] == "nn.relu":
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    relu_type = (
                        "CL_ACTIVATION_RELU" if node["name"] == "nn.relu" else "CL_ACTIVATION_RELU6"
                    )
                    self.clml_code.append(
                        self.MakeRelu.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            relu_type=relu_type,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "nn.batch_norm":
                    bn_attrs = tuple(node["attrs"]["axis"])
                    axis = int(bn_attrs[0][0])
                    bn_shape = [1, 1, 1, 1]
                    bn_node = self.nodes[node["inputs"][0][0]]
                    bn_shape[axis] = bn_node["attrs"]["shape"][0][0]
                    dtype = bn_node["attrs"]["dtype"][0][0]
                    bn_scale_tensor = get_tensor_from_map(
                        node["inputs"][0][0], shape=str(tuple(bn_shape))[1:-1], dtype=dtype
                    )
                    bn_bias_tensor = get_tensor_from_map(
                        node["inputs"][1][0], shape=str(tuple(bn_shape))[1:-1], dtype=dtype
                    )
                    bn_mean_tensor = get_tensor_from_map(
                        node["inputs"][2][0], shape=str(tuple(bn_shape))[1:-1], dtype=dtype
                    )
                    bn_var_tensor = get_tensor_from_map(
                        node["inputs"][3][0], shape=str(tuple(bn_shape))[1:-1], dtype=dtype
                    )

                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)

                    self.clml_code.append(
                        self.MakeBN.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            bn_scale_tensor=bn_scale_tensor,
                            bn_bias_tensor=bn_bias_tensor,
                            bn_mean_tensor=bn_mean_tensor,
                            bn_var_tensor=bn_var_tensor,
                            bn_attrs=str(bn_attrs)[1:-1],
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] in ["nn.max_pool2d", "nn.avg_pool2d", "nn.l2_pool2d"]:
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    pool_size = str(tuple(int(x) for x in node["attrs"]["pool_size"][0]))[1:-1]
                    strides = str(tuple(int(x) for x in node["attrs"]["strides"][0]))[1:-1]
                    padding = str(tuple(int(x) for x in node["attrs"]["padding"][0]))[1:-1]
                    self.clml_code.append(
                        self.MakePool2D.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            pool_type=node["name"],
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] in ["nn.global_max_pool2d", "nn.global_avg_pool2d"]:
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    in_node = self.nodes[node["inputs"][0][0]]
                    in_shape = str(tuple(in_node["attrs"]["shape"][0][0]))[1:-1]
                    self.clml_code.append(
                        self.MakeGlobalPool2D.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            in_shape=in_shape,
                            pool_type=node["name"],
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "reshape":
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    self.clml_code.append(
                        self.MakeReshape.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "concatenate":
                    input_len = len(node["inputs"])
                    in_list = str(
                        [get_tensor_from_map(node["inputs"][x][0]) for x in range(input_len)]
                    )[1:-1]
                    node_out_name = make_output_tensor(node, node_seq)
                    axis = node["attrs"]["axis"][0][0]
                    self.clml_code.append(
                        self.MakeConcatenate.substitute(
                            in_list=in_list,
                            output_tensor=node_out_name,
                            axis=axis,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "nn.dense":
                    in_node = self.nodes[node["inputs"][0][0]]
                    in_shape = tuple(in_node["attrs"]["shape"][0][0])
                    wt_shape = tuple(in_node["attrs"]["shape"][0][0])
                    input_tensor = get_tensor_from_map(
                        node["inputs"][0][0], layout="CL_TENSOR_LAYOUT_NCHW_QCOM"
                    )
                    weight_tensor = get_tensor_from_map(
                        node["inputs"][1][0],
                        shape=str(tuple([1, 1, wt_shape[0], wt_shape[1]]))[1:-1],
                        layout="CL_TENSOR_LAYOUT_NCHW_QCOM",
                    )
                    node_out_name = make_output_tensor(
                        node,
                        node_seq,
                        shape=str(tuple([in_shape[0], wt_shape[0], 1, 1]))[1:-1],
                        layout="CL_TENSOR_LAYOUT_NCHW_QCOM",
                    )
                    self.clml_code.append(
                        self.MakeDense.substitute(
                            input_tensor=input_tensor,
                            weight_tensor=weight_tensor,
                            output_tensor=node_out_name,
                            in_shape=str(in_shape)[1:-1],
                            wt_shape=str(wt_shape)[1:-1],
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "nn.softmax":
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    self.clml_code.append(
                        self.MakeSoftMax.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "nn.pad":
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    pad_mode = node["attrs"]["pad_mode"][0][0]
                    padding = str(tuple(int(x) for x in node["attrs"]["pad_width"][0]))[1:-1]
                    self.clml_code.append(
                        self.MakePad.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            pad_mode=pad_mode,
                            padding=padding,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "nn.batch_flatten":
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    self.clml_code.append(
                        self.MakeBatchFlatten.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] == "clip":
                    input_tensor = get_tensor_from_map(node["inputs"][0][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    a_max = node["attrs"]["a_max"][0][0]
                    a_min = node["attrs"]["a_min"][0][0]
                    self.clml_code.append(
                        self.MakeClip.substitute(
                            input_tensor=input_tensor,
                            output_tensor=node_out_name,
                            a_max=a_max,
                            a_min=a_min,
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                elif node["name"] in [
                    "add",
                    "subtract",
                    "multiply",
                    "minimum",
                    "maximum",
                    "divide",
                ]:
                    input_a = get_tensor_from_map(node["inputs"][0][0])
                    input_b = get_tensor_from_map(node["inputs"][1][0])
                    node_out_name = make_output_tensor(node, node_seq)
                    self.clml_code.append(
                        self.MakeBinaryOp.substitute(
                            input_a=input_a,
                            input_b=input_b,
                            output_tensor=node_out_name,
                            op=node["name"],
                            dtype=node["attrs"]["dtype"][0][0],
                        )
                    )
                else:
                    raise RuntimeError("Unsupported Op:" + node["name"])
                self.clml_code.append(
                    self.MapInsert.substitute(nid=node_out_name, tensor_desc=node_out_name)
                )
                self.node_map[node_seq] = node_out_name

            elif node["op"] not in ["const", "input"]:
                print("Unknown Node type:", node["op"])

        # Populate outputs
        out_nodes = self.codegen["heads"]
        self.clml_code.append("// Populate outputs")
        for nid_triple in out_nodes:
            nid = nid_triple[0]
            out_node = self.nodes[nid]
            dtype = str(out_node["attrs"]["dtype"][0][0])
            shape = str(tuple(out_node["attrs"]["shape"][0][0]))[1:-1]
            out_name = self.sub_module_name + "_" + "layer_out_" + str(nid)
            self.clml_code.append(
                Template(
                    'runner.outputs.insert({"$out_name", runner.storage_map["$out_name"]});'
                ).substitute(out_name=out_name)
            )
            self.clml_code.append(
                Template('runner.outputs_dtypes.insert({"$out_name", "$dtype"});').substitute(
                    out_name=out_name, dtype=dtype
                )
            )
            self.clml_code.append(
                Template(
                    "runner.outputs_shapes.insert" '({"$out_name", std::vector<size_t>({$shape})});'
                ).substitute(out_name=out_name, shape=shape)
            )
            self.output_meta.append(
                self.MakeOutputMetaInfo.substitute(out_name=out_name, dtype=dtype, shape=shape)
            )

        # Mem allocation & Param copy
        self.clml_code.append("// Allocate Tensor Memory and copy params")
        self.clml_code.append("runner.AllocateMemAndPopulateParams();")

        # Meta data preparation
        self.clml_code.append(
            self.MakeMetaInfo.substitute(
                name=self.sub_module_name,
                input_count=len(self.input_meta),
                output_count=len(self.output_meta),
                input_meta="\\\n".join(self.input_meta),
                output_meta="\\\n".join(self.output_meta),
            )
        )

        self.clml_code.append(self.MakeFooter.substitute())
        return (self.sub_module_name, self.clml_code)


class CLMLGenSrc:
    """Generates CLML API source given a TVM compiled mod"""

    def __init__(self, libm):
        """Initialize
        Parameters
        ----------
        libm : Module
            Compiled relay module
        """
        self.libm = libm
        self.gen_src = []
        self.clml_modules = None
        self.clml_builds = {}
        self.codegen = None
        self.nodes = None

        self.MakeFileHeader = Template(
            """/*
        * Licensed to the Apache Software Foundation (ASF) under one
        * or more contributor license agreements.  See the NOTICE file
        * distributed with this work for additional information
        * regarding copyright ownership.  The ASF licenses this file
        * to you under the Apache License, Version 2.0 (the
        * "License"); you may not use this file except in compliance
        * with the License.  You may obtain a copy of the License at
        *
        *   http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing,
        * software distributed under the License is distributed on an
        * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
        * KIND, either express or implied.  See the License for the
        * specific language governing permissions and limitations
        * under the License.
        */

        /*!
         * \\file clml_models.cc
         * \\brief CLML models for all subgraph in given TVM module.
         */

        // AUTO GENERATED BY TOOL (clml_codegen.py), PLEASE DO NOT CHANGE THIS FILE!
        // =========================================================================

        #include <iostream>
        #include <fstream>

        #include <vector>
        #include <string>
        #include <algorithm>
        #include <math.h>
        #include <list>

        // Project includes
        #include "CL/cl.h"
        #include "CL/cl_qcom_ml_ops.h"

        #include "clml_runner.h"

        using namespace tvm::runtime;
        """
        )

    def get_clml_params(self):
        """Returns parameters from the TVM module"""

        clml_params = {}
        if self.libm.get_lib().type_key == "const_loader":
            params = self.libm.get_lib().get_function("get_const_var_ndarray")()
            clml_params.update(params)

        for mod in self.libm.get_lib().imported_modules:
            if mod.type_key == "const_loader":
                params = mod.get_const_var_ndarray()
                clml_params.update(params)

        clml_params_save = {}
        for key, val in clml_params.items():
            clml_params_save[str(key)] = val.numpy()

        return clml_params_save

    def get_artifacts(self):
        """Function that returns params as dict and source as list of cource code lines"""

        self.clml_modules = list(
            filter(lambda mod: mod.type_key == "clml", self.libm.get_lib().imported_modules)
        )
        self.clml_builds["file_header"] = [self.MakeFileHeader.substitute()]

        for cmod in self.clml_modules:
            (sub_module_name, clml_code) = CLMLGetSubModuleSrc(cmod).get_src()
            self.clml_builds[sub_module_name] = clml_code

        main_code = []
        main_code.append(
            """
            std::vector<CLMLRunner> BuildModules(ToolArgs& args,
                                                 cl_platform_id arg_platform,
                                                 cl_context arg_context,
                                                 cl_device_id arg_device_id,
                                                 cl_command_queue arg_queue) {
                  std::vector<CLMLRunner> runners;"""
        )
        for key, val in self.clml_builds.items():
            if key != "file_header":
                main_code.append(
                    "runners.push_back("
                    + key
                    + '("'
                    + key
                    + '", args, arg_platform, arg_context, arg_device_id, arg_queue));'
                )
        main_code.append("return runners;}")
        self.clml_builds["MainBuild"] = main_code

        for key, val in self.clml_builds.items():
            self.gen_src.extend(val)

        return (self.get_clml_params(), self.gen_src)
