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

# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=no-else-return, no-else-continue
"""Caffe frontend."""
import numpy as np
import tvm
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .common import ExprTable
from .common import infer_shape as _infer_shape

__all__ = ["from_caffe"]


class OperatorConverter(object):
    """Operator Converted for converting Caffe ops to Relay ops"""

    def __init__(self, init_layer_dict, predict_layer, exp_tab):
        self.init_layer_dict = init_layer_dict
        self.predict_layer = predict_layer
        self.exp_tab = exp_tab
        self.new_bn = {}
        self.changed_layers = None

        self.convert_map = {
            "BatchNorm": self.convert_batch_norm,
            "Concat": self.convert_concat,
            "Convolution": self.convert_conv,
            "Crop": self.convert_crop,
            "Deconvolution": self.convert_deconv,
            "Dropout": self.convert_dropout,
            "Eltwise": self.convert_eltwise,
            "Embed": self.convert_embed,
            "Flatten": self.convert_flatten,
            "InnerProduct": self.convert_innerproduct,
            "Input": None,
            "LRN": self.convert_lrn,
            "Permute": self.convert_permute,
            "Pooling": self.convert_pooling,
            "Power": self.convert_power,
            "PReLU": self.convert_prelu,
            "ReLU": self.convert_relu,
            "Reshape": self.convert_reshape,
            "Scale": self.convert_scale,
            "Sigmoid": self.convert_sigmoid,
            "Slice": self.convert_slice,
            "Softmax": self.convert_softmax,
            "TanH": self.convert_tanh,
            "Reduction": self.convert_reduction,
        }

    def convert_flatten(self, op):
        """Convert Flatten layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])

        flatten_params = op.flatten_param.axis
        assert flatten_params == 1, "flatten axis should be 1"
        out = _op.nn.batch_flatten(in_expr)

        return out

    def convert_eltwise(self, op):
        """Convert Eltwise layer"""
        inputs = op.bottom
        assert len(inputs) >= 2, "input tensors length should be larger than 2"

        # gethering initial 2 input expressions
        lhs_expr = self.exp_tab.get_expr(inputs[0])
        rhs_expr = self.exp_tab.get_expr(inputs[1])
        lhs_shape = _infer_shape(lhs_expr)
        rhs_shape = _infer_shape(rhs_expr)
        assert lhs_shape == rhs_shape, "input tensors shape should be equal"

        eltwise_params = op.eltwise_param
        eltwise_type_dict = ["PROD", "SUM", "MAX"]
        eltwise_type = eltwise_params.operation
        coeff = list(eltwise_params.coeff)

        if eltwise_type_dict[eltwise_type] == "PROD":
            out = _op.multiply(lhs_expr, rhs_expr)
            # for rest inputs
            for i in range(len(inputs) - 2):
                extra_expr = self.exp_tab.get_expr(inputs[i + 2])
                assert _infer_shape(out) == _infer_shape(extra_expr)
                out = _op.multiply(out, extra_expr)
        elif eltwise_type_dict[eltwise_type] == "SUM":
            if coeff:
                left_coeff_expr = self.exp_tab.new_const(np.asarray(coeff[0], np.float32))
                right_coeff_expr = self.exp_tab.new_const(np.asarray(coeff[1], np.float32))
                lhs_expr_scale = _op.multiply(lhs_expr, left_coeff_expr)
                rhs_expr_scale = _op.multiply(rhs_expr, right_coeff_expr)
                out = _op.add(lhs_expr_scale, rhs_expr_scale)
            else:
                out = _op.add(lhs_expr, rhs_expr)
            # for rest inputs
            for i in range(len(inputs) - 2):
                extra_expr = self.exp_tab.get_expr(inputs[i + 2])
                assert _infer_shape(out) == _infer_shape(extra_expr)
                if coeff:
                    coeff_expr = self.exp_tab.new_const(np.asarray(coeff[i + 2], np.float32))
                    extra_expr_scale = _op.multiply(extra_expr, coeff_expr)
                    out = _op.add(out, extra_expr_scale)
                else:
                    out = _op.add(out, extra_expr)
        elif eltwise_type_dict[eltwise_type] == "MAX":
            out = _op.maximum(lhs_expr, rhs_expr)
            # for rest inputs
            for i in range(len(inputs) - 2):
                extra_expr = self.exp_tab.get_expr(inputs[i + 2])
                assert _infer_shape(out) == _infer_shape(extra_expr)
                out = _op.maximum(out, extra_expr)
        else:
            raise tvm.error.OpNotImplemented(
                "eltwise_type {} is not supported for frontend Caffe.".format(eltwise_type)
            )

        return out

    def _parse_conv_params(self, op):
        """Parse the parameters of Convolution and Deconvolution layer"""
        nonzone = lambda val, pos, dflt: val[pos] if pos < len(val) else dflt

        conv_params = op.convolution_param

        params = dict()
        # parse kernel size
        if conv_params.kernel_h > 0 or conv_params.kernel_w > 0:
            params["kernel_size"] = (conv_params.kernel_h, conv_params.kernel_w)
        else:
            ksize_h = nonzone(conv_params.kernel_size, 0, 1)
            ksize_w = nonzone(conv_params.kernel_size, 1, ksize_h)
            params["kernel_size"] = (ksize_h, ksize_w)

        # parse padding size
        if conv_params.pad_h > 0 or conv_params.pad_w > 0:
            params["padding"] = (conv_params.pad_h, conv_params.pad_w)
        else:
            pad_h = nonzone(conv_params.pad, 0, 0)
            pad_w = nonzone(conv_params.pad, 1, pad_h)
            params["padding"] = (pad_h, pad_w)

        # parse stride size
        if conv_params.stride_h > 0 or conv_params.stride_w > 0:
            params["strides"] = (conv_params.stride_h, conv_params.stride_w)
        else:
            stride_h = nonzone(conv_params.stride, 0, 1)
            stride_w = nonzone(conv_params.stride, 1, stride_h)
            params["strides"] = (stride_h, stride_w)

        # parse dilation size
        if hasattr(conv_params, "dilation") and len(conv_params.dilation) > 0:
            dilation = " ".join(str(d) for d in conv_params.dilation)
            dilation = tuple(map(int, dilation.split(" ")))
            params["dilation"] = dilation
            if len(dilation) == 1:
                params["dilation"] = (dilation[0], dilation[0])

        params["kernel_layout"] = "OIHW"
        params["data_layout"] = "NCHW"
        params["groups"] = conv_params.group
        params["channels"] = conv_params.num_output
        return params

    def convert_batch_norm(self, op):
        """Convert BatchNorm layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        n, c, h, w = _infer_shape(in_expr)

        if op.name in self.new_bn:
            mean, var, eps, gamma, beta = self.new_bn[op.name]
            mean_expr = self.exp_tab.new_const(mean, dtype="float32")
            var_expr = self.exp_tab.new_const(var, dtype="float32")
            gamma_expr = self.exp_tab.new_const(gamma, dtype="float32")
            beta_expr = self.exp_tab.new_const(beta, dtype="float32")
            out = _op.nn.batch_norm(
                in_expr, gamma_expr, beta_expr, mean_expr, var_expr, epsilon=eps, scale=True
            )

        else:
            weight_bias_blobs = self.init_layer_dict[op.name].blobs
            mean = np.asarray(weight_bias_blobs[0].data, np.float32)
            var = np.asarray(weight_bias_blobs[1].data, np.float32)
            if len(weight_bias_blobs) == 2:
                mean = np.repeat(mean, h * w).reshape((c, h, w))
                mean = np.expand_dims(mean, 0).repeat(n, axis=0)
                mean_expr = self.exp_tab.new_const(mean, dtype="float32")

                var = np.repeat(var, h * w).reshape((c, h, w))
                var = np.expand_dims(var, 0).repeat(n, axis=0)
                var_expr = self.exp_tab.new_const(var, dtype="float32")

                tmp_out = _op.multiply(in_expr, mean_expr)
                out = _op.add(tmp_out, var_expr)

                return out
            else:
                scale = np.asarray(weight_bias_blobs[2].data, np.float32)
                if scale:
                    scale = 1 / scale
            mean_expr = self.exp_tab.new_const(mean * scale, dtype="float32")
            var_expr = self.exp_tab.new_const(var * scale, dtype="float32")

            # caffe bn layer not support scale
            gamma_expr = self.exp_tab.new_const(
                np.ones(mean.shape, dtype=np.float32), dtype="float32"
            )
            beta_expr = self.exp_tab.new_const(
                np.zeros(mean.shape, dtype=np.float32), dtype="float32"
            )

            bn_params = op.batch_norm_param.eps
            out = _op.nn.batch_norm(
                in_expr, gamma_expr, beta_expr, mean_expr, var_expr, epsilon=bn_params, scale=False
            )

        return out[0]

    def convert_scale(self, op):
        """Convert Scale layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        weight_bias_blobs = self.init_layer_dict[op.name].blobs

        params = dict()
        params["bias"] = op.scale_param.bias_term
        params["axis"] = op.scale_param.axis

        gamma = np.asarray(weight_bias_blobs[0].data, np.float32)
        gamma_expr = self.exp_tab.new_const(gamma, dtype="float32")
        if params["bias"]:
            beta = np.asarray(weight_bias_blobs[1].data, np.float32)
            beta_expr = self.exp_tab.new_const(beta, dtype="float32")
        else:
            beta_expr = self.exp_tab.new_const(
                np.zeros(gamma.shape, dtype=np.float32), dtype="float32"
            )

        _, c, _, _ = _infer_shape(in_expr)
        gamma_expr = _op.reshape(gamma_expr, newshape=(1, c, 1, 1))
        beta_expr = _op.reshape(beta_expr, newshape=(1, c, 1, 1))
        out = _op.multiply(in_expr, gamma_expr)
        out = _op.add(out, beta_expr)

        return out

    def convert_concat(self, op):
        """Convert Concat layer"""
        inputs = op.bottom
        in_expr = (self.exp_tab.get_expr(inputs[i]) for i in range(len(inputs)))

        c_params = dict()
        c_params["axis"] = op.concat_param.axis
        out = _op.concatenate(in_expr, axis=c_params["axis"])

        return out

    def convert_reshape(self, op):
        """Convert Reshape layer"""
        inputs = op.bottom
        input_name = inputs[0]

        reshape_param = op.reshape_param
        dims = list(reshape_param.shape.dim)

        in_expr = self.exp_tab.get_expr(input_name)
        input_shape = list(_infer_shape(in_expr))

        start_axis = int(reshape_param.axis)
        if start_axis < 0:
            start_axis = len(input_shape) + start_axis + 1
        num_axes = int(reshape_param.num_axes)
        end_axis = len(input_shape)
        if num_axes != -1:
            end_axis = start_axis + num_axes

        left_shape = input_shape[:start_axis]
        if end_axis == len(input_shape):
            center_shape = input_shape[start_axis:]
            right_shape = []
        else:
            center_shape = input_shape[start_axis:end_axis]
            right_shape = input_shape[end_axis:]

        for idx, dim in enumerate(dims):
            if dim == 0:
                dims[idx] = center_shape[idx]

        tmp = np.random.rand(*center_shape)
        tmp = np.reshape(tmp, dims)
        center_shape = list(tmp.shape)

        newshape = left_shape + center_shape + right_shape

        out = _op.reshape(in_expr, newshape=newshape)
        return out

    def convert_softmax(self, op):
        """Convert Softmax layer"""
        inputs = op.bottom
        assert len(inputs) == 1, "input tensors length should be 1"

        input_name = inputs[0]
        in_expr = self.exp_tab.get_expr(input_name)

        softmax_param = op.softmax_param
        parmas = {"axis": softmax_param.axis}

        out = _op.nn.softmax(in_expr, **parmas)

        return out

    def convert_conv(self, op):
        """Convert Convolution layer"""
        params = self._parse_conv_params(op)
        weight_bias_blobs = self.init_layer_dict[op.name].blobs
        conv_params = op.convolution_param
        inputs = op.bottom
        # process weight and bias blobs
        weight, bias = None, None
        if len(weight_bias_blobs) > 1:
            weight = weight_bias_blobs[0]
            bias = weight_bias_blobs[1]
        else:
            weight = weight_bias_blobs[0]
        if weight:
            kh, kw = params["kernel_size"]
            weight_shape = [conv_params.num_output, -1, kh, kw]
            weight_value = np.asarray(weight.data, np.float32)
            weight_value = np.reshape(weight_value, weight_shape)
        else:
            raise Exception("No weight value of layer {} in caffemodel".format(op.name))

        weight_expr = self.exp_tab.new_const(weight_value, dtype="float32")
        in_expr = self.exp_tab.get_expr(inputs[0])
        out = _op.nn.conv2d(data=in_expr, weight=weight_expr, **params)
        if bias:
            bias_value = np.asarray(bias.data, np.float32)
            bias_expr = self.exp_tab.new_const(bias_value, dtype="float32")
            out = _op.nn.bias_add(out, bias_expr)
        return out

    def convert_pooling(self, op):
        """Convert Pooling layer"""
        inputs = op.bottom
        input_name = inputs[0]

        pool_params = op.pooling_param
        pool_type_dict = ["MAX", "AVE", "STOCHASTIC"]

        params = dict()
        # parse pool type: 0: MAX, 1: AVE, 2: STOCHASTIC
        pool_type = pool_params.pool
        # parse kernel size
        if pool_params.kernel_h > 0 or pool_params.kernel_w > 0:
            params["pool_size"] = (pool_params.kernel_h, pool_params.kernel_w)
        else:
            params["pool_size"] = (pool_params.kernel_size, pool_params.kernel_size)

        # parse padding size
        if pool_params.pad_h > 0 or pool_params.pad_w > 0:
            params["padding"] = (pool_params.pad_h, pool_params.pad_w)
        else:
            params["padding"] = (pool_params.pad, pool_params.pad)

        # parse stride size
        if pool_params.stride_h > 0 or pool_params.stride_w > 0:
            params["strides"] = (pool_params.stride_h, pool_params.stride_w)
        else:
            params["strides"] = (pool_params.stride, pool_params.stride)

        params["ceil_mode"] = True
        if hasattr(pool_params, "round_mode"):
            params["ceil_mode"] = pool_params.round_mode == "CEIL"

        in_expr = self.exp_tab.get_expr(input_name)

        if pool_type_dict[pool_type] == "MAX":
            if pool_params.global_pooling:
                out = _op.nn.global_max_pool2d(in_expr)
            else:
                if len(op.top) == 1:
                    out = _op.nn.max_pool2d(in_expr, **params)
                elif len(op.top) == 2:
                    out1 = _op.nn.max_pool2d_with_argmax(in_expr, **params)
                    out2 = _op.vision.max_pool2d_location(in_expr, **params)
                    return _expr.Tuple((out1, out2))

        elif pool_type_dict[pool_type] == "AVE":  # AVE
            if pool_params.global_pooling:
                out = _op.nn.global_avg_pool2d(in_expr)
            else:
                params["count_include_pad"] = True
                out = _op.nn.avg_pool2d(in_expr, **params)
        else:
            raise tvm.error.OpNotImplemented(
                "Operator {} is not supported for frontend Caffe.".format(
                    pool_type_dict[pool_type] + " pool"
                )
            )

        return out

    def convert_lrn(self, op):
        """Convert LRN layer"""
        inputs = op.bottom
        input_name = inputs[0]

        params = dict()
        lrn_params = op.lrn_param
        params["size"] = lrn_params.local_size
        params["bias"] = lrn_params.k
        params["alpha"] = lrn_params.alpha
        params["beta"] = lrn_params.beta

        in_expr = self.exp_tab.get_expr(input_name)
        out = _op.nn.lrn(in_expr, **params)
        return out

    def convert_innerproduct(self, op):
        """Convert InnerProduct layer"""
        inputs = op.bottom
        weight_bias_blobs = self.init_layer_dict[op.name].blobs
        dense_params = op.inner_product_param

        params = dict()
        params["num_output"] = dense_params.num_output
        params["bias"] = dense_params.bias_term
        params["axis"] = dense_params.axis
        if params["axis"] != 1:
            raise Exception("Only support 2D InnerProduct")

        # process weight and bias blobs
        weight, bias = None, None
        if params["bias"]:
            weight = weight_bias_blobs[0]
            bias = weight_bias_blobs[1]
        else:
            weight = weight_bias_blobs[0]

        if weight:
            weight_value = np.asarray(weight.data, np.float32)
            weight_value = np.reshape(weight_value, (params["num_output"], -1))
            weight_shape = weight_value.shape
        else:
            raise Exception("No weight value of layer {} in caffemodel".format(op.name))

        weight_expr = self.exp_tab.new_const(weight_value, dtype="float32")

        in_expr = self.exp_tab.get_expr(inputs[0])
        in_reshape = _op.reshape(data=in_expr, newshape=(-1, weight_shape[-1]))

        out = _op.nn.dense(data=in_reshape, weight=weight_expr)

        if bias:
            bias_value = np.asarray(bias.data, np.float32)
            bias_expr = self.exp_tab.new_const(bias_value, dtype="float32")
            out = _op.nn.bias_add(out, bias_expr, axis=params["axis"])
        return out

    def convert_dropout(self, op):
        """Convert Dropout layer"""
        inputs = op.bottom
        input_name = inputs[0]

        params = dict()
        dropout_params = op.dropout_param

        params["rate"] = dropout_params.dropout_ratio

        in_expr = self.exp_tab.get_expr(input_name)
        out = _op.nn.dropout(in_expr, **params)
        return out

    def convert_relu(self, op):
        """Convert ReLU layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        negative_slope = op.relu_param.negative_slope
        if negative_slope:
            out = _op.nn.leaky_relu(in_expr, negative_slope)
            return out

        out = _op.nn.relu(in_expr)
        return out

    def convert_prelu(self, op):
        """Convert PReLU layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])

        alpha = self.init_layer_dict[op.name].blobs[0].data
        alpha = np.asarray(alpha, np.float32)
        alpha = self.exp_tab.new_const(alpha, dtype="float32")
        axis = 1
        out = _op.nn.prelu(in_expr, alpha, axis=axis)
        return out

    def convert_deconv(self, op):
        """Convert Deconvolution layer"""
        params = self._parse_conv_params(op)
        weight_bias_blobs = self.init_layer_dict[op.name].blobs
        conv_params = op.convolution_param
        inputs = op.bottom

        # process weight and bias blobs
        weight, bias = None, None
        if len(weight_bias_blobs) > 1:
            weight = weight_bias_blobs[0]
            bias = weight_bias_blobs[1]
        else:
            weight = weight_bias_blobs[0]
        if weight:
            kh, kw = params["kernel_size"]
            weight_shape = [-1, conv_params.num_output, kh, kw]
            if not weight.data:
                if conv_params.weight_filler:
                    _filler = conv_params.weight_filler.value
                    weight_value = np.full(weight.shape.dim, _filler, np.float32)
                else:
                    raise tvm.error.OpAttributeInvalid("At least weight_filler must be given")
            else:
                weight_value = np.asarray(weight.data, np.float32)
            weight_value = np.reshape(weight_value, weight_shape)

            # weight shape is in relay's IOHW format rn, we need it to be OIHW
            weight_value = np.transpose(weight_value, [1, 0, 2, 3])
        else:
            raise tvm.error.OpAttributeRequired(
                "No weight value of layer {} in caffemodel".format(op.name)
            )

        weight_expr = self.exp_tab.new_const(weight_value, dtype="float32")
        in_expr = self.exp_tab.get_expr(inputs[0])

        groups = params["groups"]
        channels = params["channels"]

        if bias:
            bias_value = np.asarray(bias.data, np.float32)
            bias_expr = self.exp_tab.new_const(bias_value, dtype="float32")

        if groups > channels:
            raise tvm.error.OpAttributeInvalid(
                "Groups cannot be larger than the number of input channels"
            )

        if groups == channels:
            inputs_expr = _op.split(in_expr, groups, axis=1)
            # changing split axis to 0, according to PR #9336
            weights_expr = _op.split(weight_expr, groups, axis=0)
            # Preventing to create Concat layer with too many tensors(> 16)
            q = groups >> 4
            r = groups % 16

            params["groups"] = 1
            params["channels"] = 1
            out = []
            for lc in range(q):
                _outputs = []
                _inputs = [inputs_expr[i] for i in range(lc << 4, (lc << 4) + 16)]
                _weights = [weights_expr[i] for i in range(lc << 4, (lc << 4) + 16)]
                for (i, w) in zip(_inputs, _weights):
                    _out = _op.nn.conv2d_transpose(data=i, weight=w, **params)
                    if bias:
                        _out = _op.nn.bias_add(_out, bias_expr)
                    _outputs.append(_out)
                out.append(_op.concatenate(_outputs, axis=1))
            if r != 0:
                _outputs = []
                _inputs = [inputs_expr[i] for i in range(groups - r, groups)]
                _weights = [weights_expr[i] for i in range(groups - r, groups)]
                for (i, w) in zip(_inputs, _weights):
                    _out = _op.nn.conv2d_transpose(data=i, weight=w, **params)
                    if bias:
                        _out = _op.nn.bias_add(_out, bias_expr)
                    _outputs.append(_out)
                out.append(_op.concatenate(_outputs, axis=1))
            out = _op.concatenate(out, axis=1)
        elif groups == 1:
            out = _op.nn.conv2d_transpose(data=in_expr, weight=weight_expr, **params)
            if bias:
                out = _op.nn.bias_add(out, bias_expr)
        else:
            raise tvm.error.OpAttributeInvalid("Unable to handle.")
        return out

    def convert_slice(self, op):
        """Convert Slice layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])

        output_num = len(op.top)

        slice_params = op.slice_param
        axis = int(slice_params.axis)
        indices_or_sections = list([int(s) for s in slice_params.slice_point])
        if len(indices_or_sections) == 0:
            indices_or_sections = output_num
        else:
            indices_or_sections = sorted(indices_or_sections)

        out = _op.split(in_expr, indices_or_sections=indices_or_sections, axis=axis)
        return out

    def convert_sigmoid(self, op):
        """Convert Sigmoid layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        out = _op.sigmoid(in_expr)
        return out

    def convert_tanh(self, op):
        """Convert TanH layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        out = _op.tanh(in_expr)
        return out

    def convert_reduction(self, op):
        """Convert Reduction layer"""
        reduction_dic = ["NOP", "SUM", "ASUM", "SUMSQ", "MEAN"]

        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        method = op.reduction_param.operation
        axis = op.reduction_param.axis
        coeff = op.reduction_param.coeff
        coeff_expr = self.exp_tab.new_const(np.asarray(coeff, np.float32))
        num_axes = len(_infer_shape(in_expr))

        # Currently, only reduction along ALL "tail" axes is supported in Caffe;
        # reduction of axis M through N, where N < num_axes - 1, is unsupported.
        if 0 < axis < (num_axes - 1):
            for _axis in reversed(range(axis + 1, num_axes)):
                in_expr = _op.sum(in_expr, axis=_axis)
            in_expr = _op.squeeze(in_expr)

        if reduction_dic[method] == "SUM":
            out = _op.sum(in_expr, axis=axis)
        elif reduction_dic[method] == "MEAN":
            out = _op.mean(in_expr, axis=axis)
        elif reduction_dic[method] == "ASUM":
            in_expr = _op.abs(in_expr)
            out = _op.sum(in_expr, axis=axis)
        elif reduction_dic[method] == "SUMSQ":
            in_expr = _op.multiply(in_expr, in_expr)
            out = _op.sum(in_expr, axis=axis)
        else:
            raise tvm.error.OpAttributeInvalid(
                "reduction method:{} is invalid in Caffe frontend.".format(method)
            )

        if float(coeff) != 1.0:
            out = _op.multiply(out, coeff_expr)
        return out

    def convert_crop(self, op):
        """Convert Crop layer"""
        inputs = op.bottom
        assert len(inputs) == 2, "Need two inputs of Crop layer"
        in_expr_a = self.exp_tab.get_expr(inputs[0])
        in_expr_b = self.exp_tab.get_expr(inputs[1])

        # parse crop params
        crop_params = op.crop_param
        axis = int(getattr(crop_params, "axis", 2))
        offset = list(getattr(crop_params, "offset", 0))

        # expand offset to (offset1, offset2, ...)
        in_a_shape = _infer_shape(in_expr_a)
        num_to_crop = len(in_a_shape) - axis
        if not offset:
            offset = [0] * num_to_crop
        if len(offset) == 1:
            offset = offset * num_to_crop
        elif len(offset) != num_to_crop:
            raise Exception("No matching the number between axis and offset!")

        slice_end = in_a_shape
        slice_start = [0] * len(in_a_shape)
        for i in range(num_to_crop):
            slice_start[i + axis] = offset[i]

        to_crop_axis = list(range(len(in_a_shape)))
        to_crop_axis = to_crop_axis[axis:]

        # secondly, crop in_expr_a by in_expr_b
        in_expr_a_stride = _op.strided_slice(in_expr_a, slice_start, slice_end)
        out = _op.slice_like(in_expr_a_stride, in_expr_b, axes=to_crop_axis)
        return out

    def convert_permute(self, op):
        """Convert Permute layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])

        # parse permute params
        permute_param = op.permute_param
        axes = list(getattr(permute_param, "order", 0))
        out = _op.transpose(in_expr, axes)
        return out

    def convert_embed(self, op):
        """Convert Embed layer"""
        inputs = op.bottom
        embed_param = op.embed_param
        num_output = embed_param.num_output
        input_dim = embed_param.input_dim
        bias_term = embed_param.bias_term
        weight_bias_blobs = self.init_layer_dict[op.name].blobs
        weight, bias = None, None
        if bias_term:
            weight = weight_bias_blobs[0]
            bias = weight_bias_blobs[1]
            assert weight and bias
        else:
            weight = weight_bias_blobs[0]
            assert weight
        weight_value = np.asarray(weight.data, np.float32)
        weight_value = np.reshape(weight_value, [input_dim, num_output])
        weight_expr = self.exp_tab.new_const(weight_value, dtype="float32")
        in_expr = self.exp_tab.get_expr(inputs[0])
        input_shape = _infer_shape(in_expr)
        input_count = 1
        for dim in input_shape:
            input_count *= dim

        index = _op.cast(in_expr, "int32")
        out = _op.take(weight_expr, index, axis=0)

        if bias_term:
            bias_value = np.asarray(bias.data, np.float32)
            bias_expr = self.exp_tab.new_const(bias_value, dtype="float32")
            out = _op.reshape(out, [input_count, num_output])
            out = _op.add(out, bias_expr)

        out_shape = list(input_shape)
        out_shape.append(num_output)
        out = _op.reshape(out, out_shape)

        return out

    def convert_power(self, op):
        """Convert Power layer"""
        inputs = op.bottom
        in_expr = self.exp_tab.get_expr(inputs[0])
        power = _expr.const(op.power_param.power)
        scale = _expr.const(op.power_param.scale)
        shift = _expr.const(op.power_param.shift)

        out = _op.multiply(in_expr, scale)
        out = _op.add(out, shift)
        out = _op.power(out, power)
        return out

    def check_unsupported_ops(self):
        """Check unsupported Caffe ops in our converter."""
        unsupported_ops_set = set()

        include_layer = dict()
        for pl in self.predict_layer:
            if pl.type not in include_layer:
                include_layer[pl.type] = 1
            else:
                include_layer[pl.type] = include_layer[pl.type] + 1

        for pl in self.predict_layer:
            op_name = pl.type
            if op_name not in self.convert_map:
                unsupported_ops_set.add(op_name)

        if unsupported_ops_set:
            msg = "The following operators are not supported in frontend " "Caffe: {}"
            ops = str(list(unsupported_ops_set)).strip("[,]")
            raise tvm.error.OpNotImplemented(msg.format(ops))

    def fuse_op(self, layers):
        """Fusing the BatchNorm and Scale layer"""
        bn, scale = layers["bn"], layers["scale"]

        # bn params
        bn_weight_bias_blobs = self.init_layer_dict[bn.name].blobs
        bn_scale = np.asarray(bn_weight_bias_blobs[2].data, np.float32)
        if bn_scale:
            bn_scale = 1 / bn_scale
        bn_mean = np.asarray(bn_weight_bias_blobs[0].data, np.float32) * bn_scale
        bn_var = np.asarray(bn_weight_bias_blobs[1].data, np.float32) * bn_scale
        bn_eps = bn.batch_norm_param.eps

        # scale params
        scale_weight_bias_blobs = self.init_layer_dict[scale.name].blobs
        scale_gamma = np.asarray(scale_weight_bias_blobs[0].data, np.float32)
        scale_bias = scale.scale_param.bias_term
        if scale_bias:
            scale_beta = np.asarray(scale_weight_bias_blobs[1].data, np.float32)
        else:
            scale_beta = np.zeros(scale_gamma.shape, dtype=np.float32)

        # new params
        self.new_bn[bn.name] = [bn_mean, bn_var, bn_eps, scale_gamma, scale_beta]
        return bn

    def op_fuse(self):
        """fuse bn and scale"""
        new_layers = []
        temp_layers = {}
        changed_layers = {}

        for index, pl in enumerate(self.predict_layer):
            op_type = pl.type
            if op_type == "Input":
                new_layers.append(pl)
                continue
            elif op_type == "BatchNorm":
                if (index != len(self.predict_layer) - 1) and (
                    self.predict_layer[index + 1].type == "Scale"
                ):
                    temp_layers["bn"] = pl
                    continue
                else:
                    new_layers.append(pl)
                    temp_layers.clear()
            elif op_type == "Scale":
                if self.predict_layer[index - 1].type == "BatchNorm":
                    temp_layers["scale"] = pl
                else:
                    new_layers.append(pl)
                    temp_layers.clear()
            else:
                temp_layers.clear()

            if len(temp_layers) == 2:
                layer = self.fuse_op(temp_layers)
                new_layers.append(layer)
                changed_layers[temp_layers["scale"].name] = temp_layers["bn"].name

            for idx, plt in enumerate(pl.bottom):
                if plt in changed_layers:
                    pl.bottom[idx] = changed_layers[plt]

            if op_type not in ["BatchNorm", "Scale"]:
                new_layers.append(pl)

        self.predict_layer = new_layers
        self.changed_layers = changed_layers

    def convert_op_to_relay(self):
        """Convert Caffe ops to relay ops"""
        for pl in self.predict_layer:
            op_type = pl.type
            if op_type == "Input":
                continue
            output_tensors = pl.top

            ret = self.convert_map[op_type](pl)

            if len(output_tensors) == 1:
                self.exp_tab.set_expr(output_tensors[0], ret)
            else:
                for idx, output_tensor in enumerate(output_tensors):
                    self.exp_tab.set_expr(output_tensor, ret[idx])


def _rebuild_layers(predict_layer):
    """Rebuild caffe layer. If the the caffe net include in-place layers, repalce its top
    with its name and update the bottom of other layer that is related to it.
    """
    # dict of input name that will be changed to new name
    changed_top_dict = dict()

    for pl in predict_layer:
        if pl.type == "Input":
            continue
        # if current layer has single input and output and input equals to output
        # it means that the layer does "in-place"
        if len(pl.top) == 1 and len(pl.bottom) == 1:
            if pl.top[0] == pl.bottom[0]:
                # change current layer's input firstly
                if pl.bottom[0] in changed_top_dict:
                    pl.bottom[0] = changed_top_dict[pl.bottom[0]]
                # update "change" dict
                changed_top_dict[pl.top[0]] = pl.name
                # change current layer's output to its name
                pl.top[0] = pl.name
            else:
                if pl.bottom[0] in changed_top_dict:
                    pl.bottom[0] = changed_top_dict[pl.bottom[0]]
        # if the layer does not
        else:
            for index, plt in enumerate(pl.bottom):
                if plt in changed_top_dict:
                    pl.bottom[index] = changed_top_dict[plt]


def _get_inputs_outputs(predict_layer):
    """Obtain Caffe model's inputs and outpus"""
    # model inputs / outputs
    model_inputs = list()
    model_outputs = list()

    # The bottoms of every layer can not be as outputs
    not_outputs = set()
    for pl in predict_layer:
        if pl.type == "Input":
            assert len(pl.top) == 1, "The number of Input layer's output is more than 1."
            model_inputs.append(pl.top[0])
        for i in pl.bottom:
            not_outputs.add(i)

    for pl in predict_layer:
        if len(pl.bottom) > 0:
            for t in pl.top:
                if t not in not_outputs:
                    model_outputs.append(t)
    return model_inputs, model_outputs


def from_caffe(init_net, predict_net, shape_dict, dtype_dict):
    """Convert from caffe model into compatible relay Function.

    Parameters
    ----------
    init_net : caffe_pb2.NetParameter
        caffemodel
    predict_net : caffe_pb2.NetParameter
        caffe prototxt
    shape_dict : dict of str to int list/tuple
        Input shapes of the model.
    dtype_dict : dict of str to str
        Input types of the model.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    """
    old_caffe = False
    if len(predict_net.input) != 0:  # old caffe version
        old_caffe = True
        model_inputs = list(predict_net.input)

    predict_layer = predict_net.layer

    # replace layer's top with its name and update other layers'bottoms
    _rebuild_layers(predict_layer)
    # obtain inputs and outputs of Net
    if old_caffe:
        _, model_outputs = _get_inputs_outputs(predict_layer)
    else:
        model_inputs, model_outputs = _get_inputs_outputs(predict_layer)

    exp_tab = ExprTable()
    for in_name in model_inputs:
        shape = shape_dict[in_name] if in_name in shape_dict else None
        dtype = dtype_dict[in_name] if in_name in dtype_dict else "float32"
        exp_tab.set_expr(in_name, _expr.var(in_name, shape=shape, dtype=dtype))
    if list(init_net.layer):
        init_layer = init_net.layer
    else:
        init_layer = init_net.layers
    init_layer_dict = {il.name: il for il in init_layer}
    # op code in model
    op_converter = OperatorConverter(init_layer_dict, predict_layer, exp_tab)
    op_converter.check_unsupported_ops()
    op_converter.op_fuse()
    op_converter.convert_op_to_relay()

    # params and outputs
    params = {k: _nd.array(np.array(v)) for k, v in exp_tab.params.items()}
    outputs = list()
    for n in model_outputs:
        if n in op_converter.changed_layers:
            n = op_converter.changed_layers[n]
        outputs.append(exp_tab.get_expr(n))
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    func = _function.Function(analysis.free_vars(outputs), outputs)
    mod = IRModule.from_expr(func)

    return mod, params
