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
""" A set of passes to legalize some of operations for the NPU"""
import numpy as np

import tvm
from tvm import relay
from tvm import ir
from tvm.relay.dataflow_pattern import DFPatternCallback
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import is_op
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.backend.contrib.ethosu import op as ethosu_ops
from tvm.relay.backend.contrib.ethosu.errors import UnsupportedLayout
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.op.contrib import ethosu as ethosu_patterns


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
    def get_section_begin_coords(split):
        """Currently, the split operator takes an array of indices or an integer
        indicating the number of splits. However, its an array of indices could
        represent both cases, therefore this function just make it an array of
        indices where each index represent the co-ordinate of beginning of each
        section -- defines as section begins.

        Parameters
        ----------
        split : relay.Expr
            The Relay Call expression for a split operator

        Returns
        -------
        section_begins : list
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
        section_begins = list(range(0, split_axis_len, section_length))
        return section_begins

    def callback(self, pre, post, node_map):
        splits_types = dict()
        split_input = post.args[0]
        for idx, field_type in enumerate(post.checked_type.fields):
            split = relay.TupleGetItem(post, idx)
            splits_types[split] = field_type

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

    def transform_module(self, mod, ctx):
        for gv, func in mod.functions.items():
            func = rewrite(SplitRewriter(), func)
            mod.update_func(gv, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class EthosUConv2DRewriter(DFPatternCallback):
    """Convert conv2d related composite functions into ethosu_conv2d operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": "ethosu.qnn_conv2d"}))(wildcard())

    def callback(self, pre, post, node_map):
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
class LegalizeEthosUConv2D:
    """This is the pass that wraps the EthosUConv2DRewriter"""

    def transform_module(self, mod, ctx):
        for gv, func in mod.functions.items():
            func = rewrite(EthosUConv2DRewriter(), func)
            mod.update_func(gv, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeEthosU:
    """This is the pass to call graph-rewrites to perform graph transformation
    in a way such that the operations are replaced with hardware/codegen supported
    operations.
    """

    def transform_module(self, mod, ctx):
        mod = LegalizeSplit()(mod)
        mod = LegalizeEthosUConv2D()(mod)
        return mod
