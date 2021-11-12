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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel, no-value-for-parameter
"""A set of passes to legalize some of operations for the NPU"""
from typing import List, Type

import numpy as np  # type: ignore

import tvm  # type: ignore
from tvm import relay
from tvm import ir
from tvm.relay.dataflow_pattern import DFPatternCallback  # type: ignore
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import is_op
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.dataflow_pattern import CallPattern
from tvm.relay.backend.contrib.ethosu import op as ethosu_ops  # type: ignore
from tvm.relay.backend.contrib.ethosu.errors import UnsupportedLayout  # type: ignore
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.op.contrib import ethosu as ethosu_patterns  # type: ignore


class SplitRewriter(DFPatternCallback):
    """This rewriting converts split operations into a sequence of
    strided_slice operations, because codegen is going to be based
    on strided_slices that will define the slice of the tensor that
    will be fed to the consumer.
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.split_in = wildcard()
        self.pattern = is_op("split")(self.split_in)

    @staticmethod
    def get_section_begin_coords(split: tvm.relay.Expr) -> List[int]:
        """Currently, the split operator takes an array of indices or an integer
        indicating the number of splits. However, its an array of indices could
        represent both cases, therefore this function just make it an array of
        indices where each index represent the co-ordinate of beginning of each
        section -- defines as section begins.

        Parameters
        ----------
        split : tvm.relay.Expr
            The Relay Call expression for a split operator

        Returns
        -------
        section_begins : List[int]
            A list containing integers corresponding to section
            begins
        """
        indices_or_sections = split.attrs.indices_or_sections
        input_shape = split.args[0].checked_type.shape
        split_axis = split.attrs.axis

        if isinstance(indices_or_sections, tvm.ir.container.Array):
            # 0 is the beginning of the first section.
            return [0] + list(indices_or_sections)
        split_axis_len = input_shape[split_axis].value
        section_length = split_axis_len // indices_or_sections.value
        return list(range(0, split_axis_len, section_length))

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        split_input = post.args[0]
        split_begins = list()
        split_ends = list()
        section_begins_in_split_axis = self.get_section_begin_coords(post)
        for split_cord in section_begins_in_split_axis:
            # first begin is [0, 0, ... , 0]
            begin_shape = [0 for i in range(len(split_input.checked_type.shape))]
            begin_shape[post.attrs.axis] = split_cord
            split_begins.append(begin_shape)

            end_shape = list(split_input.checked_type.shape)
            # Only the split axis coordinate changes
            end_shape[post.attrs.axis] = split_cord
            split_ends.append(end_shape)

        # Coordinates needs to be shifted left because beginning
        # of the next section is the end of the previous
        split_ends = split_ends[1:]
        # Last section end is the shape of the tensor itself.
        split_ends.append(list(split_input.checked_type.shape))

        strided_slices = list()
        for sb, se in zip(split_begins, split_ends):
            strided_slices.append(relay.strided_slice(split_input, sb, se))

        return relay.Tuple(strided_slices)


@ir.transform.module_pass(opt_level=1)
class LegalizeSplit:
    """This is the pass that wraps SplitRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(SplitRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class Conv2DRewriter(DFPatternCallback):
    """Convert conv2d related composite functions into ethosu_conv2d operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": "ethosu.qnn_conv2d"}))(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.QnnConv2DParams(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }
        if str(params.ofm.layout) not in channels_map.keys():
            raise UnsupportedLayout(str(params.ofm.layout))
        kernel_size_map = {
            "HWIO": params.weights.shape[0:2],
            "OHWI": params.weights.shape[1:3],
            "HWOI": params.weights.shape[0:2],
        }
        if str(params.weights.layout) not in kernel_size_map.keys():
            raise UnsupportedLayout(str(params.weights.layout))
        activation_map = {"clip": "CLIP"}
        weight_to_ohwi_transform_map = {"HWIO": [3, 0, 1, 2]}
        weights_values = params.weights.values
        weights_values_ohwi = np.transpose(
            weights_values, weight_to_ohwi_transform_map[str(params.weights.layout)]
        )
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0
        scale_bias = vela_api.pack_biases(
            biases=params.biases.tensor.data.asnumpy(),
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=params.weights.q_params.scale_f32,
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=activation in ["TANH", "SIGMOID"],
        )
        ethosu_conv2d = ethosu_ops.ethosu_conv2d(
            ifm=post.args[0],
            weight=relay.const(weights_values_ohwi, params.weights.values.dtype),
            scale_bias=relay.const(scale_bias, "uint8"),
            lut=relay.const([], dtype="int8"),
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            weight_zero_point=int(params.weights.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            kernel_shape=kernel_size_map[str(params.weights.layout)],
            ofm_channels=params.ofm.shape[channels_map[str(params.ofm.layout)]],
            strides=params.strides,
            padding=params.padding,
            dilation=params.dilation,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
        )
        return ethosu_conv2d


@ir.transform.module_pass(opt_level=1)
class LegalizeConv2D:
    """This is the pass that wraps the Conv2DRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(Conv2DRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class DepthwiseConv2DRewriter(DFPatternCallback):
    """Convert ethosu.qnn_depthwise_conv2d composite functions to ethosu_depthwise_conv2d
    operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr(
                {"Composite": ethosu_patterns.QnnDepthwiseConv2DParams.composite_name}
            )
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.QnnDepthwiseConv2DParams(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }
        if str(params.ofm.layout) not in channels_map.keys():
            raise UnsupportedLayout(str(params.ofm.layout))
        kernel_shape_map = {
            "HWOI": params.weights.shape[0:2],
        }
        if str(params.weights.layout) not in kernel_shape_map.keys():
            raise UnsupportedLayout(str(params.weights.layout))

        weights_values = params.weights.values
        weights_values_ohwi = np.moveaxis(weights_values, [0, 1, 2, 3], [1, 2, 0, 3])

        activation = "NONE"
        # Activations requiring LUT is currently not supported, so setting it to an empty list
        lut = relay.const([], "int8")
        clip_min = 0
        clip_max = 0
        if params.activation:
            activation = ethosu_patterns.QnnDepthwiseConv2DParams.activation_map[
                params.activation.op.name
            ]
            if activation == "CLIP":
                clip_min = int(params.activation.attrs.a_min)
                clip_max = int(params.activation.attrs.a_max)
        scale_bias = vela_api.pack_biases(
            biases=params.biases.tensor.data.asnumpy(),
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=params.weights.q_params.scale_f32,
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=activation in ["TANH", "SIGMOID"],
        )

        ethosu_depthwise_conv2d = ethosu_ops.ethosu_depthwise_conv2d(
            post.args[0],  # IFM
            relay.const(weights_values_ohwi, params.weights.values.dtype),
            relay.const(scale_bias, "uint8"),
            lut,
            float(params.ifm.q_params.scale_f32),
            int(params.ifm.q_params.zero_point),
            int(params.weights.q_params.zero_point),
            float(params.ofm.q_params.scale_f32),
            int(params.ofm.q_params.zero_point),
            kernel_shape_map[str(params.weights.layout)],
            params.ofm.shape[channels_map[str(params.ofm.layout)]],
            strides=params.strides,
            padding=params.padding,
            dilation=params.dilation,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
        )
        return ethosu_depthwise_conv2d


@ir.transform.module_pass(opt_level=1)
class LegalizeDepthwiseConv2D:
    """This is the pass that wraps the DepthwiseConv2DRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(DepthwiseConv2DRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class PoolingRewriter(DFPatternCallback):
    """Convert ethosu.avgpool2d and ethosu.maxpool2d composite functions to
    ethosu_pooling operators"""

    def __init__(
        self,
        params_class: Type,
        pattern: CallPattern,
    ):
        super().__init__(require_type=True)
        self.params_class = params_class
        self.pattern = pattern

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }
        if str(params.ofm.layout) not in channels_map.keys():
            raise UnsupportedLayout(str(params.ofm.layout))

        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        # Activations requiring LUT is currently not supported, so setting it to an empty list
        lut = relay.const([], dtype="int8")

        return ethosu_ops.ethosu_pooling(
            ifm=post.args[0],
            lut=lut,
            pooling_type=params.pooling_type,
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_zero_point=params.ifm.q_params.zero_point,
            ofm_scale=params.ofm.q_params.scale_f32,
            ofm_zero_point=params.ofm.q_params.zero_point,
            pool_shape=params.pool_shape,
            ofm_channels=params.ofm.shape[channels_map[str(params.ofm.layout)]],
            strides=params.strides,
            padding=params.padding,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
        )


class MaxPoolingRewriter(PoolingRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MaxPool2DParams,
            pattern=(
                wildcard().has_attr({"Composite": ethosu_patterns.MaxPool2DParams.composite_name})
            )(wildcard()),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeMaxPooling:
    """This is the pass that wraps the MaxPoolingRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MaxPoolingRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class AvgPoolingRewriter(PoolingRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.AvgPool2DParams,
            pattern=(
                wildcard().has_attr({"Composite": ethosu_patterns.AvgPool2DParams.composite_name})
            )(wildcard()),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeAvgPooling:
    """This is the pass that wraps the AvgPoolingRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(AvgPoolingRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class BinaryElementwiseRewriter(DFPatternCallback):
    """Convert ethosu binary elementwise composite functions to
    ethosu_binary_elementwise operators"""

    def __init__(
        self,
        params_class: Type,
        pattern: CallPattern,
    ):
        super().__init__(require_type=True)
        self.params_class = params_class
        self.pattern = pattern

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[1] if params.reversed_operands else post.args[0]
        params.ifm2.tensor = post.args[0] if params.reversed_operands else post.args[1]
        channels_map = {
            "NHWC": 3,
        }
        if str(params.ofm.layout) not in channels_map.keys():
            raise UnsupportedLayout(str(params.ofm.layout))

        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        # We don't yet support activation functions that need to get legalized to LUTs.
        lut = relay.const([], dtype="int8")

        return ethosu_ops.ethosu_binary_elementwise(
            ifm=params.ifm.tensor,
            ifm2=params.ifm2.tensor,
            lut=lut,
            operator_type=params.operator_type,
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ifm2_scale=float(params.ifm2.q_params.scale_f32),
            ifm2_zero_point=int(params.ifm2.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            ifm_channels=params.ifm.shape[3],
            ifm2_channels=params.ifm2.shape[3],
            reversed_operands=params.reversed_operands,
            ofm_dtype=params.ofm.dtype,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            ifm_layout=str(params.ifm.layout),
            ifm2_layout=str(params.ifm2.layout),
            ofm_layout=str(params.ofm.layout),
        )


class AddRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.AddParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.AddParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeAdd:
    """This is the pass that wraps the AddRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(AddRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class SubRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.SubParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.SubParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeSub:
    """This is the pass that wraps the SubRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(SubRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class MulRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MulParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.MulParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeMul:
    """This is the pass that wraps the MulRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MulRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class MinRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MinParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.MinParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeMin:
    """This is the pass that wraps the MinRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MinRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class MaxRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MaxParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.MaxParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeMax:
    """This is the pass that wraps the MaxRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MaxRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class ShlRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.ShlParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.ShlParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


@ir.transform.module_pass(opt_level=1)
class LegalizeShl:
    """This is the pass that wraps the ShlRewriter"""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(ShlRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeEthosU:
    """This is the pass to call graph-rewrites to perform graph transformation
    in a way such that the operations are replaced with hardware/codegen supported
    operations.
    """

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        """This is the method that replaces the operations with hardware/codegen supported
        operations.
        """
        mod = LegalizeSplit()(mod)
        mod = LegalizeConv2D()(mod)
        mod = LegalizeDepthwiseConv2D()(mod)
        mod = LegalizeMaxPooling()(mod)
        mod = LegalizeAvgPooling()(mod)
        mod = LegalizeAdd()(mod)
        mod = LegalizeSub()(mod)
        mod = LegalizeMul()(mod)
        mod = LegalizeMin()(mod)
        mod = LegalizeMax()(mod)
        mod = LegalizeShl()(mod)
        return mod

    def __call__(self, *args, **kwargs):
        # pylint is unable figure out the decorated
        # class is callable, thus adding this to
        # suppress the warning.
        pass
