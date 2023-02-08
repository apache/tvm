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
"""
A set of passes to legalize the Gemmini operators
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from typing import Tuple
import tvm  # type: ignore
from tvm import relay
from tvm import ir
from tvm.relay.dataflow_pattern import DFPatternCallback  # type: ignore
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import rewrite

from tvm.relay.op import _make  # type: ignore

from .pattern_table import AddParams, CONV2DParams, GEMMParams, MaxPoolParams  # type: ignore


def gemmini_gemm(
    ifm1: tvm.relay.Expr,
    ifm2: tvm.relay.Expr,
    bias: tvm.relay.Expr,
    ifm_scale: float,
    ifm_offset: float,
    bias_scale: float,
    bias_offset: float,
    ofm_scale: float,
    ofm_offset: float,
) -> tvm.relay.Call:
    """Generates the call to the contrib.gemmini.gemm operator

    Args:
        ifm1 (tvm.relay.Expr): Input feature map 1
        ifm2 (tvm.relay.Expr): Input feature map 2 (weights)
        bias (tvm.relay.Expr): Biases
        ifm_scale (float): Input feature map scaling factor
        ifm_offset (float): Input feature map offset
        bias_scale (float): Biases scaling factor
        bias_offset (float): Biases offset
        ofm_scale (float): Output feature map scaling factor
        ofm_offset (float): Output feature map offset

    Returns:
        tvm.relay.Call: Call to the contrib.gemmini.gemm operator
    """
    return _make.gemmini_gemm(
        ifm1, ifm2, bias, ifm_scale, ifm_offset, bias_scale, bias_offset, ofm_scale, ofm_offset
    )


def gemmini_add(
    ifm1: tvm.relay.Expr,
    ifm2: tvm.relay.Expr,
    ifm1_scale: float,
    ifm1_offset: float,
    ifm2_scale: float,
    ifm2_offset: float,
    ofm_scale: float,
    ofm_offset: float,
    shape: Tuple[int, ...],
) -> tvm.relay.Call:
    """Generates the call to the contrib.gemmini.add operator

    Args:
        ifm1 (tvm.relay.Expr): Input feature map 1
        ifm2 (tvm.relay.Expr): Input feature map 2
        ifm1_scale (float): Input feature map 1 scaling factor
        ifm1_offset (float): Input feature map 1 offset
        ifm2_scale (float): Input feature map 2 scaling factor
        ifm2_offset (float): Input feature map 2 offset
        ofm_scale (float): Output feature map scaling factor
        ofm_offset (float): Output feature map offset
        shape (Tuple[int,...]): Shape of the input feature maps and the output feature map

    Returns:
        tvm.relay.Call: Call to the contrib.gemmini.add operator
    """
    return _make.gemmini_add(
        ifm1,
        ifm2,
        ifm1_scale,
        ifm1_offset,
        ifm2_scale,
        ifm2_offset,
        ofm_scale,
        ofm_offset,
        shape,
    )


def gemmini_conv2d(
    data: tvm.relay.Expr,
    weights: tvm.relay.Expr,
    bias: tvm.relay.Expr,
    strides: tuple,
    padding: tuple,
    ifm_scale: float,
    ifm_offset: float,
    weights_scale: float,
    weights_offset: float,
    bias_scale: float,
    bias_offset: float,
    ofm_scale: float,
    ofm_offset: float,
    activation: bool,
    has_pool: bool,
    pool_size: tvm.relay.Expr,
    pool_strides: tvm.relay.Expr,
    pool_dilation: tvm.relay.Expr,
    pool_padding: tvm.relay.Expr,
    input_req_offset_out: tvm.relay.Expr,
    has_activation: bool,
    activation_scale_in: tvm.relay.Expr,
    activation_offset_in: tvm.relay.Expr,
    activation_scale_out: tvm.relay.Expr,
    activation_offset_out: tvm.relay.Expr,
) -> tvm.relay.Call:
    """Generates the call to the contrib.gemmini.conv2d operator

    Args:
        data (tvm.relay.Expr): Input feature map
        weights (tvm.relay.Expr): Convolution weights matrix
        bias (tvm.relay.Expr): Convolution biases matrix
        strides (tuple): Convolution strides
        padding (tuple): Convolution paddings in each direction
        ifm_scale (float): Input feature map scaling factor
        ifm_offset (float): Input feature map offset
        weights_scale (float): Weights scaling factor
        weights_offset (float): Convolution weights offset
        bias_scale (float): Biases scaling factor
        bias_offset (float): Biases weights offset
        ofm_scale (float): Output feature map scaling factor
        ofm_offset (float): Output feature map offset
        activation (bool): TODO (FP): see if this can be deleted! Has activation?
        has_pool (bool): Has pooling layer after the output of the convolution?
        pool_size (tvm.relay.Expr): Pooling window size
        pool_strides (tvm.relay.Expr): Pooling window strides
        pool_dilation (tvm.relay.Expr): Pooling window dilation
        pool_padding (tvm.relay.Expr): Pooling padding in each direction
        input_req_offset_out (tvm.relay.Expr): Requantize layer output offset
        has_activation (bool): Has activation?
        activation_scale_in (tvm.relay.Expr): TODO (FP): check if this can be deleted and made more simple. Activation layer input scaling factor
        activation_offset_in (tvm.relay.Expr): TODO (FP): check if this can be deleted and made more simple. Activation layer input offset
        activation_scale_out (tvm.relay.Expr): TODO (FP): check if this can be deleted and made more simple. Activation layer output scaling factor
        activation_offset_out (tvm.relay.Expr): TODO (FP): check if this can be deleted and made more simple. Activation layer output offset

    Returns:
        tvm.relay.Call: Call to the contrib.gemmini.conv2d operator
    """
    return _make.gemmini_conv2d(
        data,
        weights,
        bias,
        strides,
        padding,
        ifm_scale,
        ifm_offset,
        weights_scale,
        weights_offset,
        bias_scale,
        bias_offset,
        ofm_scale,
        ofm_offset,
        activation,
        has_pool,
        pool_size,
        pool_strides,
        pool_dilation,
        pool_padding,
        input_req_offset_out,
        has_activation,
        activation_scale_in,
        activation_offset_in,
        activation_scale_out,
        activation_offset_out,
    )


def gemmini_depthwise_conv2d(
    data: tvm.relay.Expr,
    weights: tvm.relay.Expr,
    bias: tvm.relay.Expr,
    strides: tuple,
    padding: tuple,
    ifm_scale: float,
    ifm_offset: float,
    weights_scale: float,
    weights_offset: float,
    bias_scale: float,
    bias_offset: float,
    ofm_scale: float,
    ofm_offset: float,
    activation: bool,
) -> tvm.relay.Call:
    """Generates the call to the contrib.gemmini.depthwiseconv2d operator

    Args:
        data (tvm.relay.Expr): Input feature map
        weights (tvm.relay.Expr): Convolution weights matrix
        bias (tvm.relay.Expr): Convolution biases matrix
        strides (tuple): Convolution strides
        padding (tuple): Convolution paddings in each direction
        ifm_scale (float): Input feature map scaling
        ifm_offset (float): Input feature map offset
        weights_scale (float): Convolution weights scaling factor
        weights_offset (float): Convolution weights offset
        bias_scale (float): Convolution biases scaling factor
        bias_offset (float): Convolution biases offset
        ofm_scale (float): Output feature map scaling
        ofm_offset (float): Output feature map offset
        activation (bool): Has activation?

    Returns:
        tvm.relay.Call: Call to the contrib.gemmini.depthwiseconv2d operator
    """
    return _make.gemmini_depthwise_conv2d(
        data,
        weights,
        bias,
        strides,
        padding,
        ifm_scale,
        ifm_offset,
        weights_scale,
        weights_offset,
        bias_scale,
        bias_offset,
        ofm_scale,
        ofm_offset,
        activation,
    )


def gemmini_max_pool2d(
    ifm: tvm.relay.Expr,
    pool_size: tvm.relay.Expr,
    pool_strides: tvm.relay.Expr,
    pool_dilation: tvm.relay.Expr,
    pool_padding: tvm.relay.Expr,
    shape: tuple,
) -> tvm.relay.Call:
    """Generates the call to the contrib.gemmini.max_pool2d operator

    Args:
        ifm (tvm.relay.Expr): Input feature map
        pool_size (tvm.relay.Expr): Pooling window size
        pool_strides (tvm.relay.Expr): Pooling window strides
        pool_dilation (tvm.relay.Expr): Pooling window dilation
        pool_padding (tvm.relay.Expr): Pooling padding in each direction
        shape (tuple): Input shape

    Returns:
        tvm.relay.Call: Call to the contrib.gemmini.max_pool2d operator
    """
    return _make.gemmini_max_pool2d(
        ifm, pool_size, pool_strides, pool_dilation, pool_padding, shape
    )


class AddRewriter(DFPatternCallback):
    """Convert add related composite functions into contrib.gemmini.add operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": AddParams.composite_name}))(
            wildcard(), wildcard()
        )

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = AddParams(post.op.body)
        gemmini_add_op = gemmini_add(
            post.args[0],
            post.args[1],
            params.ifm1_scale,
            params.ifm1_offset,
            params.ifm2_scale,
            params.ifm2_offset,
            params.ofm_scale,
            params.ofm_offset,
            params.output_shape,
        )
        return gemmini_add_op


class GEMMRewriter(DFPatternCallback):
    """Convert gemm related composite functions into contrib.gemmini.gemm operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": GEMMParams.composite_name}))(
            wildcard(), wildcard(), wildcard()
        )

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = GEMMParams(post.op.body)
        gemmini_gemm_op = gemmini_gemm(
            post.args[0],
            post.args[1],
            post.args[2],
            params.ifm_scale,
            params.ifm_offset,
            params.bias_scale,
            params.bias_offset,
            params.ofm_scale,
            params.ofm_offset,
        )
        return gemmini_gemm_op


class CONV2DRewriter(DFPatternCallback):
    """Convert conv2d related composite functions into contrib.gemmini.conv2d operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": CONV2DParams.composite_name}))(
            wildcard(), wildcard(), wildcard()
        )
        self.data_index = 0
        self.weights_index = 1
        self.bias_index = 2

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = CONV2DParams(post.op.body)
        if params.has_external_pad:
            self.weights_index = 2
            self.bias_index = 3
        else:
            self.weights_index = 1
            self.bias_index = 2

        bias = post.args[self.bias_index]

        if params.has_input_requantize:
            data = relay.cast(post.args[self.data_index], "int8")
        else:
            data = post.args[self.data_index]

        if params.is_depthwise:
            reshaped_weights = relay.squeeze(
                relay.transpose(post.args[self.weights_index], [3, 0, 1, 2]), axis=[3]
            )
            gemmini_depthwise_conv2d_op = gemmini_depthwise_conv2d(
                data=data,
                weights=reshaped_weights,
                bias=bias,
                strides=params.strides,
                padding=params.padding,
                ifm_scale=params.ifm_scale,
                ifm_offset=params.ifm_offset,
                weights_scale=params.weights_scale,
                weights_offset=params.weights_offset,
                bias_scale=params.bias_scale,
                bias_offset=params.bias_offset,
                ofm_scale=params.ofm_scale,
                ofm_offset=params.ofm_offset,
                activation=params.activation,
            )
            return gemmini_depthwise_conv2d_op
        else:
            gemmini_conv2d_op = gemmini_conv2d(
                data=data,
                weights=post.args[self.weights_index],
                bias=bias,
                strides=params.strides,
                padding=params.padding,
                ifm_scale=params.ifm_scale,
                ifm_offset=params.ifm_offset,
                weights_scale=params.weights_scale,
                weights_offset=params.weights_offset,
                bias_scale=params.bias_scale,
                bias_offset=params.bias_offset,
                ofm_scale=params.ofm_scale,
                ofm_offset=params.ofm_offset,
                activation=params.activation,
                has_pool=params.has_pool,
                pool_size=params.pool_size,
                pool_strides=params.pool_strides,
                pool_dilation=params.pool_dilation,
                pool_padding=params.pool_padding,
                input_req_offset_out=params.input_offset_out,
                has_activation=params.has_activation,
                activation_scale_in=params.activation_scale_in,
                activation_offset_in=params.activation_offset_in,
                activation_scale_out=params.activation_scale_out,
                activation_offset_out=params.activation_offset_out,
            )
        return gemmini_conv2d_op


class CONV2DExternalPadRewriter(CONV2DRewriter):
    def __init__(self):
        super().__init__()
        self.pattern = (wildcard().has_attr({"Composite": CONV2DParams.composite_name}))(
            wildcard(), wildcard(), wildcard(), wildcard()
        )
        self.data_index = 0


class CONV2DExternalPadAndRelu6Rewriter(CONV2DRewriter):
    def __init__(self):
        super().__init__()
        self.pattern = (wildcard().has_attr({"Composite": CONV2DParams.composite_name}))(
            wildcard(), wildcard(), wildcard(), wildcard(), wildcard()
        )
        self.data_index = 0
        self.min_index = 4


class MAXPOOL2DRewriter(DFPatternCallback):
    """Convert conv2d related composite functions into gemmini_max_pool2d operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": MaxPoolParams.composite_name}))(
            wildcard()
        )
        self.data_index = 0

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = MaxPoolParams(post.op.body)

        data = post.args[self.data_index]

        gemmini_max_pool2d_op = gemmini_max_pool2d(
            ifm=data,
            pool_size=params.pool_size,
            pool_strides=params.pool_strides,
            pool_dilation=params.pool_dilation,
            pool_padding=params.pool_padding,
            shape=params.shape,
        )
        return gemmini_max_pool2d_op


@ir.transform.module_pass(opt_level=1)
class LegalizeAdd:
    """This is the pass that wraps the AddRewriter"""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(AddRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeMaxPool2D:
    """This is the pass that wraps the MAXPOOL2DRewriter"""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MAXPOOL2DRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeGEMM:
    """This is the pass that wraps the GEMMRewriter"""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(GEMMRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeCONV2D:
    """This is the pass that wraps the CONV2DRewriter"""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(CONV2DRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeCONV2DExternalPad:
    """This is the pass that wraps the CONV2DExternalPadRewriter"""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(CONV2DExternalPadRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeCONV2DExternalPadAndRelu6:
    """This is the pass that wraps the CONV2DExternalPadAndRelu6Rewriter"""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(CONV2DExternalPadAndRelu6Rewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


@ir.transform.module_pass(opt_level=1)
class LegalizeGemmini:
    """This is the pass to call graph-rewrites to perform graph transformation
    in a way such that the operations are replaced with hardware/codegen supported
    operations.
    """

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        """This is the method that replaces the operations with hardware/codegen supported
        operations.
        """
        mod = LegalizeCONV2DExternalPadAndRelu6()(mod)
        mod = LegalizeCONV2DExternalPad()(mod)
        mod = LegalizeAdd()(mod)
        mod = LegalizeCONV2D()(mod)
        mod = LegalizeGEMM()(mod)
        mod = LegalizeMaxPool2D()(mod)
        return mod

    def __call__(self, *args, **kwargs):
        # pylint is unable figure out the decorated
        # class is callable, thus adding this to
        # suppress the warning.
        pass
