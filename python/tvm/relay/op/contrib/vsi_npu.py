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

import tvm.ir
import numpy as np
from tvm.relay import transform
from ...dataflow_pattern import wildcard, is_op, is_constant, is_tuple_get_item
from .register import register_pattern_table
from ... import qnn as _qnn
from . import vsi_npu_ffi_api as support_api
from tvm.relay.build_module import bind_params_by_name

@register_pattern_table("vsi_npu")
def vsi_npu_pattern_table():

    def qnn_conv_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern :
            Denotes the convolution pattern.
        """
        pattern = is_op("qnn.conv2d")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant())))
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_avg_pool2d_pattern():
        """Create a quantized average pool pattern.
        Returns
        -------
        pattern :
            Denotes the quant-average pool pattern.
        """
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def qnn_adaptive_avg_pool2d_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.adaptive_avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def qnn_softmax_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.softmax")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_sigmoid_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("sigmoid")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_clip_pattern():
        pattern = is_op("clip")(wildcard())
        pattern = pattern.optional(lambda x: (is_op("qnn.requantize")(
            x, is_constant(), is_constant(), is_constant(), is_constant()
        )))
        return pattern

    def qnn_leaky_relu_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.leaky_relu")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_tanh_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("tanh")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_dense_pattern():
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant())))
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_mean_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("mean")(pattern)
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant())
        return pattern

    def qnn_deconv_pattern():
        pattern = is_op("qnn.conv2d_transpose")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def dropout_pattern():
        pattern = is_op("nn.dropout")(wildcard())
        pattern = is_tuple_get_item(pattern)
        return pattern

    vsi_npu_patterns = [
            ("vsi_npu.qnn_deconv", qnn_deconv_pattern()),
            ("vsi_npu.qnn_dense",qnn_dense_pattern()),
            ("vsi_npu.qnn_conv2d", qnn_conv_pattern()),
            ("vsi_npu.qnn_avgpool2d", qnn_avg_pool2d_pattern()),
            ("vsi_npu.qnn_adaptive_avg_pool2d", qnn_adaptive_avg_pool2d_pattern()),
            ("vsi_npu.qnn_softmax", qnn_softmax_pattern()),
            ("vsi_npu.qnn_sigmoid", qnn_sigmoid_pattern()),
            ("vsi_npu.qnn_clip", qnn_clip_pattern()),
            ("vsi_npu.qnn_mean", qnn_mean_pattern()),
            ("vsi_npu.qnn_leaky_relu", qnn_leaky_relu_pattern()),
            ("vsi_npu.qnn_tanh", qnn_tanh_pattern()),
            ("vsi_npu.dropout", dropout_pattern()),
            ]
    return vsi_npu_patterns

def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.vsi_npu")
    def _func_wrapper(args):
        return supported

    return _func_wrapper

_register_external_op_helper("qnn.add")
_register_external_op_helper("qnn.subtract")
_register_external_op_helper("qnn.mul")
_register_external_op_helper("maximum")
_register_external_op_helper("minimum")
_register_external_op_helper("logical_and")
_register_external_op_helper("logical_or")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("mean")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("reshape")
_register_external_op_helper("squeeze")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("clip")
_register_external_op_helper("qnn.quantize")
_register_external_op_helper("qnn.dequantize")
_register_external_op_helper("qnn.requantize")
_register_external_op_helper("qnn.concatenate")
_register_external_op_helper("image.resize2d")
_register_external_op_helper("argmax")
_register_external_op_helper("argmin")
_register_external_op_helper("transpose")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.depth_to_space")
_register_external_op_helper("nn.pad")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("nn.conv2d_transpose")


class QnnQuantizeConstFold(tvm.relay.dataflow_pattern.DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.pattern = is_op("qnn.quantize")(
            is_constant(), is_constant(), is_constant()
        )

    def callback(self, pre, post, node_map):
        data = pre.args[0].data.numpy()
        scale = pre.args[1].data.numpy()
        zp = pre.args[2].data.numpy()
        data = np.around(data / scale + zp)
        dtype = pre.checked_type.dtype
        if (dtype == "int8"):
            return tvm.relay.Constant(tvm.nd.array(data.astype(np.int8)))
        if (dtype == "int32"):
            return tvm.relay.Constant(tvm.nd.array(data.astype(np.int32)))
        return post

class RemoveClipAfterRequantize(tvm.relay.dataflow_pattern.DFPatternCallback):
    def __init__(self, require_type=True):
        super().__init__(require_type)
        requantize = is_op("qnn.requantize")(
           wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        clip = is_op("clip")(requantize)
        self.pattern = is_op("cast")(clip)

    def callback(self, pre, post, node_map):
        requantize = post.args[0].args[0]
        dtype = post.attrs.dtype
        if (requantize.attrs.out_dtype == "int32") and (dtype == "uint8"):
            return tvm.relay.qnn.op.requantize(
                requantize.args[0],
                requantize.args[1],  # input scale
                requantize.args[2],  # input zero point
                requantize.args[3],  # output scale
                requantize.args[4],  # output zero point
                out_dtype=dtype,
                axis = requantize.attrs.axis,
                rounding = requantize.attrs.rounding
                )
        else:
            return post

def partition_for_vsi_npu(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to VSI NPU.

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
            transform.RemoveUnusedFunctions(),
            transform.FoldConstant(),
        ]
    )
    mod = seq(mod)

    func = mod["main"]
    func = tvm.relay.Function(func.params, func.body, None, func.type_params, func.attrs)

    func = tvm.relay.dataflow_pattern.rewrite(RemoveClipAfterRequantize(), func)
    func = tvm.relay.dataflow_pattern.rewrite(QnnQuantizeConstFold(), func)

    mod = tvm.IRModule.from_expr(func)
    # print(mod.astext(show_meta_data=False))

    seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(vsi_npu_pattern_table()),
            transform.AnnotateTarget("vsi_npu"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    return seq(mod)
