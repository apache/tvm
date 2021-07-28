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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""Paddle: PArallel Distributed Deep LEarning."""
import copy
import warnings
import six

import numpy as np

import tvm
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .common import (
    AttrCvt,
    fold_constant,
    infer_channels,
    infer_shape,
    infer_type,
    infer_value,
    new_var,
)

__all__ = ["from_paddle"]

def is_fixed_shape(shape):
    for s in shape:
        if s < 0:
            return False
    return True

def convert_feed(g, op, block):
    """Converter for model input node."""

    ipt_name = op.output('Out')[0]
    ipt_shape = block.var(ipt_name).shape
    ipt_dtype = block.var(ipt_name).dtype
    ipt_dtype = str(ipt_dtype).strip().split('.')[1]
    if g.shape_dict is not None:
        ipt_shape = g.shape_dict[ipt_name]
    print("88888", ipt_name, ipt_shape)

    g.nodes[ipt_name] = new_var(ipt_name, shape=ipt_shape, dtype=ipt_dtype)

def convert_scale(g, op, block):
    """Operator converter for scale."""

    scale = op.attr('scale')
    bias = op.attr('bias')
    bias_after_scale = op.attr('bias_after_scale')
    x = g.nodes[op.input('X')[0]]
    if np.isclose(scale, 1.0) and np.isclose(bias, 0.0):
        out = _op.copy(x)
    else:
        if np.isclose(bias, 0.0):
            out = x * _expr.const(np.array(scale).astype('float32'))
        elif np.isclose(scale, 1.0):
            out = x + _expr.const(np.array(bias).astype('float32'))
        else:
            if bias_after_scale:
                out = x * _expr.const(np.array(scale).astype('float32')) + _expr.const(np.array(bias).astype('float32'))
            else:
                out = (x + _expr.const(np.array(bias).astype('float32'))) * _expr.const(np.array(scale).astype('float32'))
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_nearest_interp_v2(g, op, block):
    """Operator converter for nearest_interp_v2."""
    
    align_corners = op.attr('align_corners')
    align_mode = op.attr('align_mode')
    scale = op.attr('scale')
    layout = op.attr('data_layout')
    assert layout == 'NCHW', "Only NCHW is supported for PaddlePaddle's nearest_interp_v2"
    assert len(scale) == 2, "scale should contain 2 value for PaddlePaddle's nearest_interp_v2"
    x = g.nodes[op.input('X')[0]]
    x_shape = infer_shape(x)
    out = _op.nn.upsampling(x, scale[0], scale[1], layout='NCHW', method='nearest_neighbor', align_corners=align_corners)    
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_bilinear_interp_v2(g, op, block):
    """Operator converter for bilinear_interp_v2."""
    
    align_corners = op.attr('align_corners')
    align_mode = op.attr('align_mode')
    layout = op.attr('data_layout')
    assert not align_corners, "Only align_corners==False is supported for PaddlePaddle's bilinear_interp_v2"
    assert layout == 'NCHW', "Only NCHW is supported for PaddlePaddle's bilinear_interp_v2"
    x = g.nodes[op.input('X')[0]]
    outsize = g.nodes[op.input('OutSize')[0]]
    outsize = infer_value(outsize)
    
    coordinate_transformation_mode = 'asymmetric'
    if align_mode == 0:
        coordinate_transformation_mode = 'half_pixel'

    out = _op.image.resize(x, outsize, method='bilinear', coordinate_transformation_mode=coordinate_transformation_mode) 
    g.nodes[op.output('Out')[0]] = fold_constant(out)
    
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr('alpha')
    x = g.nodes[op.input('X')[0]]
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_hard_swish(g, op, block):
    """Operator converter for hard_swish."""

    offset = op.attr('offset')
    scale = op.attr('scale')
    threshold = op.attr('threshold')
    assert np.isclose(offset, 3.0), "Only support offset==3.0 for PaddlePaddle's hard_swish"
    assert np.isclose(scale, 6.0), "Only support scale==6.0 for PaddlePaddle's hard_swish"
    assert np.isclose(threshold, 6.0), "Only support threshold==6.0 for PaddlePaddle's hard_swish"
    x = g.nodes[op.input('X')[0]]
    out = _op.clip(x, -3.0, 3.0)
    out = out / _expr.const(6.0) + _expr.const(0.5)
    out = x * out
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_pool2d(g, op, block):
    """Operator converter for pool2d."""

    layout = op.attr('data_format')
    assert layout == 'NCHW', "Only support NCHW format for PaddlePaddle's pool2d."
    adaptive = op.attr('adaptive')
    ceil_mode = op.attr('ceil_mode')
    exclusive = op.attr('exclusive')
    global_pooling = op.attr('global_pooling')
    ksize = op.attr('ksize')
    paddings = op.attr('paddings')
    padding_algorithm = op.attr('padding_algorithm')
    pooling_type = op.attr('pooling_type')

    op_map = {
        'avg': 'avg_pool2d',
        'max': 'max_pool2d',
    }
    strides = op.attr('strides')
    assert exclusive, "Only support exclusive==True for PaddlePaddle's pool2d"
    assert padding_algorithm == "EXPLICIT", "Only support padding_algorithm==EXPLICIT for PaddlePaddle's pool2d"
    if isinstance(strides, int):
        strides = [strides, strides]
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(paddings, six.string_types):
        msg = "Setting paddings to `SAME` or `VALID` is not support for PaddlePaddle's pool2d"
        raise tvm.error.OpNotImplemented(msg)
    elif isinstance(paddings, int):
        paddings = [paddings] * 2
    elif len(paddings) == 2:
        pass
    elif len(paddings) == 4:
        msg = "Only support length of paddings equals to 2 for PaddlePaddle's pool2d"
        raise tvm.error.OpNotImplemented(msg)

    x = g.nodes[op.input('X')[0]]
    if not adaptive:
        out = getattr(_op.nn, op_map[pooling_type])(x, pool_size=ksize, strides=strides, padding=paddings, ceil_mode=ceil_mode)
    else:
        out = getattr(_op.nn, "adaptive_" + op_map[pooling_type])(x, output_size=ksize)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_matmul(g, op, block):
    """Operator converter for matmul."""
    op_type = op.type
    assert op.type in ["matmul", "matmul_v2"]
    x = g.nodes[op.input('X')[0]]
    y = g.nodes[op.input('Y')[0]]
    x_shape = infer_shape(x)
    y_shape = infer_shape(y)
    try:
        trans_x = op.attr('trans_x')
        trans_y = op.attr('trans_y')
    except:
        trans_x = op.attr('transpose_X')
        trans_y = op.attr('transpose_Y')
    if trans_x:
        perm = list(range(len(x_shape)))
        perm[-2] = len(x_shape) - 1
        perm[-1] = len(x_shape) - 2
        x = _op.transpose(x, axes=perm)
    if trans_y:
        perm = list(range(len(y_shape)))
        perm[-2] = len(y_shape) - 1
        perm[-1] = len(x_shape) - 2
        y = _op.transpose(y, axes=perm)
    x_shape = infer_shape(x)
    y_shape = infer_shape(y)
    if len(x_shape) > 2 or len(y_shape) > 2:
        if not is_fixed_shape(x_shape) or not is_fixed_shape(y_shape):
            msg = "Inputs have to be fixed shape while rank of input > 2 for PaddlePaddle's matmul"
            raise tvm.error.OpNotImplemented(msg)
        def flatten_to_3d(data, data_shape):
            ndims = len(data_shape)
            new_shape = [-1, data_shape[-2], data_shape[-1]]
            new_data = _op.reshape(data, new_shape)
            return new_data
        x = flatten_to_3d(x, x_shape)
        y = flatten_to_3d(y, y_shape)
        y = _op.transpose(y, [0, 2, 1])
        out = _op.nn.batch_matmul(x, y)
        if len(x_shape) > len(y_shape):
            out_batch = x_shape[0:-2]
        elif len(y_shape) > len(x_shape):
            out_batch = y_shape[0:-2]
        else:
            x_batch = x_shape[0:-2]
            y_batch = y_shape[0:-2]
            out_batch = max(x_batch, y_batch)
        final_shape = list(out_batch) + [x_shape[-2], y_shape[-1]]
        out = _op.reshape(out, final_shape)
    else:
        if len(y_shape) == 2:
            y = _op.transpose(y, axes=[1, 0])
        out = _op.dense(x, y)
    try:
        alpha = op.attr('alpha')
        if not np.isclose(alpha, 1.0):
            out = out * _expr.const(alpha).astype('float32')
    except:
        pass
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_cast(g, op, block):
    """Operator converter for cast."""

    dtype = block.var(op.output('Out')[0]).dtype
    dtype = str(dtype).strip().split('.')[1]
    x = g.nodes[op.input('X')[0]]
    out = _op.cast(x, dtype=dtype)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_hard_sigmoid(g, op, block):
    """Operator converter for hard_sigmoid."""

    slope = op.attr('slope')
    offset = op.attr('offset')
    x = g.nodes[op.input('X')[0]]
#    out = _op.clip(x, -3.0, 3.0)
    out = x * _expr.const(slope) + _expr.const(0.5)
    out = _op.clip(out, 0, 1)
#    out = out / _expr.const(6.0) + _expr.const(0.5)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_multiclass_nms3(g, op, block):
    """Operator converter for multiclass_nms3."""
    
    boxes = g.nodes[op.input('BBoxes')[0]]
    scores = g.nodes[op.input('Scores')[0]]
    assert len(infer_shape(boxes)) == 3 and len(infer_shape(scores)) == 3
    batch_size, num_boxes, num_cord = infer_shape(boxes)
    batch_size, num_classes, num_boxes = infer_shape(scores)
    assert batch_size == 1, "Only batch_size==1 is supported for PaddlePaddle's multiclass_nms3"

    nms_top_k = op.attr('nms_top_k')
    normalized = op.attr('normalized')
    background_label = op.attr('background_label')
    keep_top_k = op.attr('keep_top_k')
    nms_eta = op.attr('nms_eta')
    nms_threshold = op.attr('nms_threshold')
    score_threshold = op.attr('score_threshold')
    assert normalized, "Only normalized==True is Supported for PaddlePaddle's multiclass_nms3"

    g.nodes[op.output('Out')[0]] = fold_constant(boxes)
    g.nodes[op.output('NmsRoisNum')[0]] = fold_constant(scores)
    return

    indices, num_total_boxes = _op.vision.all_class_non_max_suppression(boxes, scores, max_output_boxes_per_class=nms_top_k, iou_threshold=nms_threshold, score_threshold=score_threshold)
    num_total_boxes = _op.cast(num_total_boxes, dtype='int32')


    num_indices, _ = infer_shape(indices)
    if nms_top_k > 0 and nms_top_k * num_classes < num_indices:
        indices = _op.strided_slice(indices, begin=[0, 0], end=[batch_size*nms_top_k*num_classes, 3])
        num_indices = nms_top_k * num_classes

    filters = np.array([i for i in range(num_indices)]).astype('int32')
    filters = filters.reshape((-1, 1))
    filters = _expr.const(filters)
    compare = _op.less(filters, num_total_boxes)
    indices = indices * compare.astype('int64')

    batch_boxid = _op.strided_slice(indices, begin=[0, 0], end=[num_indices, 3], strides=[1, 2])
    class_id = _op.strided_slice(indices, begin=[0, 1], end=[num_indices, 2])
    batch_boxid = _op.transpose(batch_boxid, axes=[1, 0])

    filter_boxes = _op.gather_nd(boxes, batch_boxid, 0)
    new_indices = _op.transpose(indices, axes=[1, 0])
    filter_scores = _op.gather_nd(scores, new_indices, 0)

    compare = _op.reshape(compare, (-1, ))
    filter_scores = filter_scores * compare.astype('float32')

    if keep_top_k > 0 and keep_top_k < num_indices:
        filter_scores, topk_idx = _op.topk(filter_scores, k=keep_top_k, ret_type='both', is_ascend=False)
        topk_idx = _op.reshape(topk_idx, [-1, keep_top_k])
        filter_boxes = _op.gather_nd(filter_boxes, topk_idx)
        class_id = _op.gather_nd(class_id, topk_idx)
        num_total_boxes = _op.minimum(num_total_boxes, _expr.const(np.array([keep_top_k]).astype('int32')))
    filter_scores = _op.reshape(filter_scores, (-1, 1))
    class_id = _op.cast(class_id, dtype='float32')
    out = _op.concatenate([class_id, filter_scores, filter_boxes], axis=1)
    g.nodes[op.output('Out')[0]] = fold_constant(out)
    g.nodes[op.output('NmsRoisNum')[0]] = fold_constant(num_total_boxes)

def convert_yolo_box(g, op, block):
    """Operator converter for yolo_box."""
    
    x = g.nodes[op.input('X')[0]]
    img_size = g.nodes[op.input('ImgSize')[0]]
    anchors = op.attr('anchors')
    class_num = op.attr('class_num')
    clip_bbox = op.attr('clip_bbox')
    conf_thresh = op.attr('conf_thresh')
    downsample_ratio = op.attr('downsample_ratio')
    scale_x_y = op.attr('scale_x_y')

    n, c, h, w = infer_shape(x)
    assert h > 0 and w > 0, "Only support fixed shape of input for PaddlePaddle's yolo_box"
    an_num = int(len(anchors) // 2)
    bias_x_y = -0.5 * (scale_x_y - 1.)
    input_h = downsample_ratio * h
    input_w = downsample_ratio * w

    x = _op.reshape(x, (n, an_num, 5+class_num, h, w))
    x = _op.transpose(x, (0, 1, 3, 4, 2))

    pred_box_xy = _op.strided_slice(x, begin=[0, 0, 0, 0, 0], end=[n, an_num, h, w, 2])
    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1)).reshape(h, w, 1).astype('float32')
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w)).reshape(w, h, 1).astype('float32')
    pred_box_xy = _op.sigmoid(pred_box_xy) * _expr.const(scale_x_y) + _expr.const(bias_x_y) + _expr.const(np.concatenate([grid_x, grid_y], axis=-1))
    pred_box_xy = pred_box_xy / _expr.const(np.array([w, h]).astype('float32').reshape([2]))

    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors_s = np.array([(an_w / input_w, an_h / input_h) for an_w, an_h in anchors])
    anchor_w = anchors_s[:, 0:1].reshape((1, an_num, 1, 1)).astype('float32')
    anchor_h = anchors_s[:, 1:2].reshape((1, an_num, 1, 1)).astype('float32')
    pred_box_wh = _op.strided_slice(x, begin=[0, 0, 0, 0, 2], end=[n, an_num, h, w, 4])
    pred_box_wh = _op.exp(pred_box_wh) * _expr.const(np.stack([anchor_w, anchor_h], axis=-1))

    pred_conf = _op.strided_slice(x, begin=[0, 0, 0, 0, 4], end=[n, an_num, h, w, 5])
    pred_conf = _op.sigmoid(pred_conf)
    filters = _op.greater_equal(pred_conf, _expr.const(conf_thresh))
    filters = _op.cast(filters, dtype='float32')
    pred_conf = pred_conf * filters
    pred_score = _op.strided_slice(x, begin=[0, 0, 0, 0, 5], end=[n, an_num, h, w, 5+class_num])
    pred_score = _op.sigmoid(pred_score) * pred_conf
    pred_score = _op.reshape(pred_score, (n, -1, class_num))

    filters = _op.cast(_op.greater(pred_conf, _expr.const(0.0)), dtype='float32')

    center = pred_box_wh / _expr.const(2.0)
    pred_box_xy_min = pred_box_xy - center
    pred_box_xy_max = pred_box_xy + center
    new_img_size = _op.reverse(img_size, axis=1)
    new_img_size = _op.cast(new_img_size, dtype='float32')
    wh_const = _op.concatenate([new_img_size, new_img_size], axis=-1)
    pred_box = _op.concatenate([pred_box_xy_min, pred_box_xy_max], axis=-1)
    pred_box = pred_box * filters
    pred_box = _op.reshape(pred_box, (n, -1, 4))
    pred_box = pred_box * wh_const

    if clip_bbox:
        n, b, _ = infer_shape(pred_box)
        pred_box_xy_min = _op.strided_slice(pred_box, begin=[0, 0, 0], end=[n, b, 2])
        pred_box_xy_max = _op.strided_slice(pred_box, begin=[0, 0, 2], end=[n, b, 4])
        pred_box_xy_min = _op.clip(pred_box_xy_min, 0, np.inf)
        max_w_h = _op.reshape(new_img_size-_expr.const(1.0), (n, 1, 2))
        pred_box_xy_max = _op.minimum(pred_box_xy_max, max_w_h)
        pred_box = _op.concatenate([pred_box_xy_min, pred_box_xy_max], axis=-1)
	
    g.nodes[op.output('Boxes')[0]] = fold_constant(pred_box)
    g.nodes[op.output('Scores')[0]] = fold_constant(pred_score)

def convert_conv2d(g, op, block):
    """Operator converter for conv2d."""

    def get_pad_size(in_size, dilated_kernel_size, stride_size):
        if stride_size == 1 or in_size & stride_size == 0:
            pad = max(dilated_kernel_size - stride_size, 0)
        else:
            pad = max(dilated_kernel_size - (in_size % stride_size), 0)
        return [pad//2, pad-pad//2]

    assert op.attr('data_format') == 'NCHW', "Only NCHW format is support for PaddlePaddle's conv2d"
    dilations = op.attr('dilations')
    groups = op.attr('groups')
    paddings = op.attr('paddings')
    padding_algorithm = op.attr('padding_algorithm')
    strides = op.attr('strides')
   
    kernel = g.nodes[op.input('Filter')[0]]
    input = g.nodes[op.input('Input')[0]]
    out_channels, _, k_h, k_w = infer_shape(kernel)
    in_h, in_w = infer_shape(input)[2:]
    assert len(paddings) == 2, "Only support len(paddings)==2 for PaddlePaddle's conv2d"
    assert len(dilations) == 2, "Only support len(dilations)==2 for PaddlePaddle's conv2d"
    if padding_algorithm == "SAME":
        pad_h = get_pad_size(in_h, (k_h-1)*dilations[0]+1, strides[0])
        pad_w = get_pad_size(in_w, (k_w-1)*dilations[1]+1, strides[1])
        paddings = [pad_h[0], pad_w[0], pad_h[1], pad_w[1]]
    out = _op.nn.conv2d(input, kernel, strides=strides, padding=paddings, dilation=dilations, groups=groups, channels=out_channels, kernel_size=[k_h, k_w])
    g.nodes[op.output('Output')[0]] = fold_constant(out)

def convert_activation(g, op, block):
    """Operator converter for all the activation."""

    op_map = {
        'exp': _op.exp,
        'relu': _op.nn.relu,
        'tanh': _op.tanh,
        'sqrt': _op.sqrt,
        'erf': _op.erf,
        'abs': _op.abs,

    }
    act_func = op_map[op.type]
    out = act_func(g.nodes[op.input('X')[0]])
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_elementwise_op(g, op, block):
    """Operator converter for all the elementwise operators."""

    op_map = {
        'elementwise_div': lambda x, y: x / y,
        'elementwise_add': lambda x, y: x + y,
        'elementwise_mul': lambda x, y: x * y,
        'elementwise_sub': lambda x, y: x - y,
        'elementwise_mod': lambda x, y: x % y,
    }
    op_func = op_map[op.type]
    ipt0 = g.nodes[op.input('X')[0]]
    ipt1 = g.nodes[op.input('Y')[0]]
    ipt0_shape = block.var(op.input('X')[0]).shape
    ipt1_shape = block.var(op.input('Y')[0]).shape
    axis = op.attr('axis')
    if len(ipt0_shape) != len(ipt1_shape):
        if axis < 0:
            axis = axis + len(ipt0_shape)
        if axis != len(ipt0_shape) - 1:
            ipt1 = _op.expand_dims(ipt1, axis=axis, num_newaxis=(len(ipt0_shape) - axis - 1))
    out = op_func(ipt0, ipt1)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_concat(g, op, block):
    """Operator converter for concat."""

    inputs = [g.nodes[op.input('X')[i]] for i in range(len(op.input('X')))]
    axis = op.attr('axis')
    out = _op.concatenate(inputs, axis=axis)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_softmax(g, op, block):
    """Operator converter for softmax."""

    axis = op.attr('axis')
    input_shape = block.var(op.input('X')[0]).shape
    if axis < 0:
        axis = len(input_shape) + axis
    x = g.nodes[op.input('X')[0]]
    m = _op.max(x, axis, keepdims=True)
    e = _op.exp(x - m)
    out = e / _op.sum(e, axis, keepdims=True)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_reshape(g, op, block):
    """Operator converter for reshape."""

    shape = op.attr('shape')
    assert len(shape) > 0, "Unexpected situation happend in convert_reshape function"
    out = _op.reshape(g.nodes[op.input('X')[0]], shape)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_assign(g, op, block): 
    """Operator converter for assign."""

    out = _op.copy(g.nodes[op.input('X')[0]])
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_batch_norm(g, op, block):
    """Operator converter for batch_norm."""

    ipt_name = op.input('X')[0]
    scale_name = op.input('Scale')[0]
    bias_name = op.input('Bias')[0]
    mean_name = op.input('Mean')[0]
    variance_name = op.input('Variance')[0]
    epsilon = op.attr('epsilon')
    momentum = op.attr('momentum')
    out = _op.nn.batch_norm(g.nodes[ipt_name], g.nodes[scale_name], g.nodes[bias_name], g.nodes[mean_name], g.nodes[variance_name], epsilon=epsilon)
    g.nodes[op.output('Y')[0]] = fold_constant(out[0])

def convert_fill_constant(g, op, block):
    """Operator converter for fill_constant."""

    value = op.attr('value')
    shape = block.var(op.output('Out')[0]).shape
    dtype = block.var(op.output('Out')[0]).dtype
    dtype = str(dtype).strip().split('.')[1]
    value = np.full(shape, value, dtype)
    out = _expr.const(value.astype(dtype)).astype(dtype)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_transpose(g, op, block):
    """Operator converter for transpose."""

    perm = op.attr('axis')
    out = _op.transpose(g.nodes[op.input('X')[0]], axes=perm)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_slice(g, op, block):
    """Operator converter for slice."""

    def parameter_process(starts, ends, axes):
        new_axes = []
        new_starts = []
        new_ends = []
        pop_index = 0
        for i in range(max(axes) + 1):
            new_axes.append(i)
            if i in axes:
                new_starts.append(starts[pop_index])
                new_ends.append(ends[pop_index])
                pop_index += 1
            else:
                new_starts.append(0)
                new_ends.append(np.iinfo(np.int32).max)
        return new_starts, new_ends, new_axes

    starts = op.attr('starts')
    ends = op.attr('ends')
    axes = op.attr('axes')
    if isinstance(starts, int):
        starts = [starts]
    if isinstance(ends, int):
        ends = [ends]
    if isinstance(axes, int):
        axes = [axes]
    starts, ends, axes = parameter_process(starts, ends, axes)
    out = _op.strided_slice(g.nodes[op.input('Input')[0]], begin=starts, end=ends)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_shape(g, op, block):
    """Operator converter for shape."""

    x = g.nodes[op.input('Input')[0]]
    shape = infer_shape(x)
    for i in shape:
        if i < 0:
            msg = "Dynamic shape is not support yet, the shape of {} is {} now.".format(op.input('Input')[0], shape)
            raise tvm.error.OpNotImplemented(msg)
    g.nodes[op.output('Out')[0]] = _expr.const(np.array(shape).astype('int32'))

def convert_arg_max(g, op, block):
    """Operator converter for arg_max."""

    axis = op.attr('axis')
    keepdims = op.attr('keepdims')
    flatten = op.attr('flatten')
    assert not flatten, "Only flatten==True is supported for PaddlePaddle's arg_max"

    x = g.nodes[x.input('X')[0]]
    out = _op.argmax(x, axis=axis, keepdims=keepdims)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_dropout(g, op, block):
    """Operator converter for dropout."""

    x = g.nodes[op.input('X')[0]]
    out = _op.copy(x)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_unsqueeze(g, op, block):
    """Operator converter for unsqueeze."""

    x = g.nodes[op.input('X')[0]]
    axes = sorted(op.attr('axes'))
    for axis in axes:
        x = _op.expand_dims(x, axis=axis, num_newaxis=1)
    g.nodes[op.output('Out')[0]] = fold_constant(x)

def convert_cumsum(g, op, block):
    """Operator converter for cumsum."""

    axis = op.attr('axis')
    exclusive = op.attr('exclusive')
    flatten = op.attr('flatten')
    reverse = op.attr('reverse')

    assert not flatten, "Only flatten==False is supported for PaddlePaddle's cumsum"

    x = g.nodes[op.input('X')[0]]
    if reverse:
        x = _op.reverse(x, axis=axis)
        out = _op.cumsum(x, axis=axis, exclusive=exclusive)
        out = _op.reverse(out, axis=axis)
    else:
        out = _op.cumsum(x, axis=axis, exclusive=exclusive)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_equal(g, op, block):
    """Operator converter for equal."""

    x = g.nodes[op.input('X')[0]]
    y = g.nodes[op.input('Y')[0]]
    out = _op.equal(x, y)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_layer_norm(g, op, block):
    """Operator converter for layer_norm."""

    begin_norm_axis = op.attr('begin_norm_axis')
    epsilon = op.attr('epsilon')
    x = g.nodes[op.input('X')[0]]
    bias = g.nodes[op.input('Bias')[0]]
    scale = g.nodes[op.input('Scale')[0]]
    out = _op.nn.layer_norm(x, gamma=scale, beta=bias, axis=begin_norm_axis, epsilon=epsilon, center=True, scale=True)
    g.nodes[op.output('Y')[0]] = fold_constant(out)

def convert_fill_any_like(g, op, block):
    """Operator converter for fill_any_like."""

    out_name = op.output('Out')[0]
    out_dtype = block.var(out_name).dtype
    out_dtype = str(out_dtype).strip().split('.')[1]
    x = g.nodes[op.input('X')[0]]
    ipt_shape = infer_shape(x)
    if not is_fixed_shape(ipt_shape):
        msg = "Only support fixed input shape of PaddlePaddle's fill_any_like"
        raise tvm.error.OpNotImplemented(msg)
    value = op.attr('value')
    const = np.ones(ipt_shape) * value
    g.nodes[op.output('Out')[0]] = _expr.const(const.astype(out_dtype))

def convert_lookup_table(g, op, block):
    """Operator converter for lookup_table_v2."""

    indices = g.nodes[op.input('Ids')[0]]
    padding_idx = op.attr('padding_idx')
    is_sparse = op.attr('is_sparse')
    height_sections = op.attr('height_sections')
    if padding_idx != -1:
        g.params[op.input('W')[0]][padding_idx] = 0.0
        g.nodes[op.input('W')[0]] = _expr.const(g.params[op.input('W')[0]])
    weights = g.nodes[op.input('W')[0]]
    out = _op.take(weights, indices.astype('int32'), axis=0)
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_gelu(g, op, block):
    """Operator converter for gelu."""

    x = g.nodes[op.input('X')[0]]
    out = x * (_expr.const(0.5, dtype='float32') +
            _op.erf(x * _expr.const(0.5**0.5, dtype='float32')) * 
            _expr.const(0.5, dtype='float32'))
    g.nodes[op.output('Out')[0]] = fold_constant(out)

def convert_range(g, op, block):
    """Operator converter for range."""

    start = g.nodes[op.input('Start')[0]]
    end = g.nodes[op.input('End')[0]]
    step = g.nodes[op.input('Step')[0]]

    out_name = op.output('Out')[0]
    out_dtype = block.var(out_name).dtype
    out_dtype = str(out_dtype).strip().split('.')[1]

    out = _op.arange(start, end, step, dtype=out_dtype)
    g.ndoes[out_name] = fold_constant(out)

def convert_reduce_op(g, op, block):
    """Operator converter for reduce_op."""

    op_map = {
        'reduce_sum': _op.reduce.sum,
        'reduce_mean': _op.reduce.mean,
        'reduce_prod': _op.reduce.prod,
    }
    assert op.type in op_map
    dim = op.attr('dim')
    keep_dim = op.attr('keep_dim')
    reduce_all = op.attr('reduce_all')

    x = g.nodes[op.input('X')[0]]
    dim = None if reduce_all else dim

    op_func = op_map[op.type]
    out = op_func(x, axis=dim, keepdims=keep_dim)
    g.nodes[op.output('Out')[0]] = fold_consant(out)

def convert_flatten_continguous_range(g, op, block):
    """Operator converter for flatten_continguous_range."""

    start_axis = op.attr('start_axis')
    stop_axis = op.attr('stop_axis')
    x = g.nodes[op.input('X')[0]]
    ipt_shape = infer_shape(x)
    
    if start_axis < 0:
        start_axis += len(ipt_shape)
    if stop_axis < 0:
        stop_axis += len(ipt_shape)

_convert_map = {
    'feed': convert_feed,
    'scale': convert_scale,
    'conv2d': convert_conv2d,
    'depthwise_conv2d': convert_conv2d,
    'exp': convert_activation,
    'relu': convert_activation,
    'softmax': convert_softmax,
    'elementwise_div': convert_elementwise_op,
    'elementwise_mul': convert_elementwise_op,
    'elementwise_add': convert_elementwise_op,
    'elementwise_sub': convert_elementwise_op,
    'concat': convert_concat,
    'reshape2': convert_reshape,
    'batch_norm': convert_batch_norm,
    'assign': convert_assign,
    'fill_constant': convert_fill_constant,
    'transpose2': convert_transpose,
    'slice': convert_slice,
    'nearest_interp_v2': convert_nearest_interp_v2,
    'leaky_relu': convert_leaky_relu,
    'yolo_box': convert_yolo_box,
    'hard_swish': convert_hard_swish,
    'cast': convert_cast,
    'hard_sigmoid': convert_hard_sigmoid,
    'pool2d': convert_pool2d,
    'multiclass_nms3': convert_multiclass_nms3,
    'shape': convert_shape,
    'arg_max': convert_arg_max,
    'bilinear_interp_v2': convert_bilinear_interp_v2,
    'matmul': convert_matmul,
    'matmul_v2': convert_matmul,
    'dropout': convert_dropout,
    'fill_any_like': convert_fill_any_like,
    'unsqueeze2': convert_unsqueeze,
    'tanh': convert_activation,
    'equal': convert_equal,
    'lookup_table_v2': convert_lookup_table,
    'layer_norm': convert_layer_norm,
    'cumsum': convert_cumsum,
    'gelu': convert_gelu,
}


class GraphProto(object):
    """ A helper class for handling relay functions from PaddlePaddle model."""

    def __init__(self):
        self.nodes = {}
        self.params = {}
        self.shape_dict = None

    def extract_parameters(self, program, scope):
        """ Extract all the weights from PaddlePaddle program."""

        self.params = {}
        variables = program.global_block().vars
        for name in variables:
            var = program.global_block().var(name)
            if name.endswith('feed') or name.endswith('fetch'):
                continue
            if not var.persistable:
                continue
            self.params[name] = np.array(scope.var(name).get_tensor())
            self.nodes[name] = _expr.const(self.params[name])

    def check_input_shape(self, op, block):
        """ Check the shape information of model's inputs, fixed shape is recommended."""

        ipt_name = op.input(op.input_names[0])
        ipt_shape = block.var(ipt_name).shape
        for i in ipt_shape:
            if i < 0:
                warning_msg = (
                    "Input {}(shape={}) has unkown dimension shapes. Specifying static values may improve performance".format(ipt_name, ipt_shape))
                warings.warn(warning_msg)
                    
    def check_unsupported_ops(self, program):
        """ Check whether all the operators are supported."""

        unsupported_ops = set()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == 'fetch':
                    continue
                if op.type not in _convert_map:
                    unsupported_ops.add(op.type)
        if len(unsupported_ops) > 0:
            msg = "The following operators are not supported for frontend Paddle: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def ops_to_relay(self, program, scope):
        """ Convert PaddlePaddle operators to TVM relay functions."""

        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == 'fetch':
                    continue
                convert_func = _convert_map[op.type]
                convert_func(self, op, block)

    def get_outputs(self, program):
        """ Get outputs of PaddlePaddle model."""

        outputs = list()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == "fetch":
                    outputs.append(op.input('X')[0])
        return outputs

    def from_paddle(self, program, shape_dict, scope):
        """ Construct the TVM relay expression from PaddlePaddle program."""

        self.shape_dict = shape_dict
        if scope is None:
            import paddle
            scope = paddle.fluid.global_scope()
        self.check_unsupported_ops(program)
        self.extract_parameters(program, scope)
        self.ops_to_relay(program, scope)
        output_names = self.get_outputs(program)

        outputs = [self.nodes[name] for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        return mod, self.params

def from_paddle(program, shape_dict=None, scope=None):
    """ Convert a PaddlePaddle model into an equivalent Relay Function.

    PaddlePaddle program represent the computation graph of PaddlePaddle model, 
    and PaddlePaddle scope stores all the weights of PaddlePaddle model. 
    """
    g = GraphProto()
    mod, params = g.from_paddle(program, shape_dict, scope)
    return mod, params
