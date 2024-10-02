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
NNEF frontend graph definitions for test cases
"""


# pylint: disable=line-too-long

gt_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = gt(input1, input2);
}
"""

max_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = max(input1, input2);
}
"""

local_contrast_normalization = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = local_contrast_normalization(input, size = [1, 1, 3, 3], bias = 1.0, epsilon = 1e-5);
}
"""

mean_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = mean_reduce(input, axes = [2,3]);
}
"""

select_4d = """
version 1.0;

graph G( cond, input1, input2 ) -> ( output )
{
    cond = external<logical>(shape = [4,16,32,32]);
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = select(cond, input1, input2);
}
"""

max_pool3x3_pad1_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,0), (1,0)], border = 'ignore');
}
"""

relu = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = relu(conv);
}
"""

atanh_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = atanh(input);
}
"""

split_channel = """
version 1.0;

graph G( input ) -> ( output1, output2 )
{
    input = external(shape = [4,16,32,32]);
    [output1, output2] = split(input, axis = 1, ratios = [1,1]);
}
"""

rcp_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = rcp(input);
}
"""

max_pool2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,2,2], stride = [1,1,2,2], border = 'ignore');
}
"""

silu_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    output = silu(input);
}
"""

avg_pool2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,2,2], stride = [1,1,2,2], border = 'constant');
}
"""

separable_deconv5x5 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    plane_filter = variable(shape = [8,1,5,5], label = 'plane_filter');
    point_filter = variable(shape = [16,8,1,1], label = 'point_filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = separable_deconv(input, plane_filter, point_filter, bias);
}
"""

slice_strides = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = slice(input, axes = [1,2,3], begin = [5,16,2], end = [1,4,-1], stride = [-1,-1,1]);
}
"""

matmul_4d_transpose = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = matmul(input1, input2, transposeA = true, transposeB = false);
}
"""

rcp_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = rcp(input);
}
"""

log2_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = log2(input);
}
"""

conv3x3_stride2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, stride = [2,2]);
}
"""

lt_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = lt(input, 0.5);
}
"""

or_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<logical>(shape = [4,16,32,32]);
    input2 = external<logical>(shape = [4,16,32,32]);
    output = or(input1, input2);
}
"""

tan_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = tan(input);
}
"""

deconv7x7 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,7,7], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

acos_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = acos(input);
}
"""

nearest_upsample = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = nearest_upsample(input, factor = [2,2]);
}
"""

ceil_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = ceil(input);
}
"""

floor_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = floor(input);
}
"""

avg_pool1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,1,1], stride = [1,1,2,2], border = 'constant');
}
"""

log_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = log(input);
}
"""

sum_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sum_reduce(input, axes = [1]);
}
"""

min_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = min_reduce(input, axes = [2,3]);
}
"""

asinh_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = asinh(input);
}
"""

max_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = max(input1, input2);
}
"""

max_pool3x3_pad0_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,1), (0,1)], border = 'ignore');
}
"""

cos_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = cos(input);
}
"""

not_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = not(input);
}
"""

sub_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = sub(input1, input2);
}
"""

bilinear_upsample_aligned_replicate = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = multilinear_upsample(input, factor = [2,2], method = 'aligned', border = 'replicate');
}
"""

log_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = log(input);
}
"""

argmin_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = argmin_reduce(input, axes = [2,3]);
}
"""

selu_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    output = selu(input);
}
"""

select_2d = """
version 1.0;

graph G( cond, input1, input2 ) -> ( output )
{
    cond = external<logical>(shape = [4,16]);
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = select(cond, input1, input2);
}
"""

prelu = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external(shape = [16,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input1, filter, bias, groups = 0);
    input2 = external(shape = [16]);
    output = prelu(conv, input2);
}
"""

ne_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = ne(input1, input2);
}
"""

or_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<logical>(shape = [4,16]);
    input2 = external<logical>(shape = [4,16]);
    output = or(input1, input2);
}
"""

eq_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = eq(input1, input2);
}
"""

rsqr_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = rsqr(input);
}
"""

eq_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = eq(input1, input2);
}
"""

deconv7x7_stride4x4 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,7,7], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, stride = [4,4]);
}
"""

max_pool3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'ignore');
}
"""

and_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<logical>(shape = [4,16,32,32]);
    input2 = external<logical>(shape = [4,16,32,32]);
    output = and(input1, input2);
}
"""

atan_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = atan(input);
}
"""

pad_0_1_reflect = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (0,1), (0,1)], border = 'reflect');
}
"""

mul_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = mul(input1, input2);
}
"""

softmax = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = softmax(conv, axes = [1]);
}
"""

sign_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sign(input);
}
"""

mul_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = mul(input, 0.5);
}
"""

le_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = le(input, 0.5);
}
"""

box2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,2,2], stride = [1,1,2,2], border = 'constant');
}
"""

or_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<logical>(shape = [4,16,32,32]);
    input2 = external<logical>(shape = [1,16,1,1]);
    output = or(input1, input2);
}
"""

deconv5x5 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,5,5], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

box3x3_pad1_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,0), (1,0)], border = 'constant');
}
"""

debox3x3_pad1_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,0), (1,0)], border = 'constant');
}
"""

ge_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = ge(input1, input2);
}
"""

linear_reshape = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,8,8]);
    weights = variable(shape = [32,1024], label = 'weights');
    bias = variable(shape = [1,32], label = 'bias');
    flattened = reshape(input, shape = [0,-1]);
    output = linear(flattened, weights, bias);
}
"""

le_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = le(input1, input2);
}
"""

deconv3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

nearest_downsample = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = nearest_downsample(input, factor = [2,2]);
}
"""

select_4d_true = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = select(true, input1, input2);
}
"""

min_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = min(input1, input2);
}
"""

max_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = max(input1, input2);
}
"""

max_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = max(input, 0.5);
}
"""

sum_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sum_reduce(input, axes = [2,3]);
}
"""

min_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = min(input1, input2);
}
"""

ge_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = ge(input1, input2);
}
"""

conv2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,2,2], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

conv4x4_stride2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,4,4], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, stride = [2,2]);
}
"""

debox1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,1,1], stride = [1,1,2,2], padding = [(0,0),(0,0),(0,-1),(0,-1)], border = 'constant');
}
"""

reshape_flatten = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = reshape(input, shape = [0,-1]);
}
"""

conv3x3_nobias = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    output = conv(input, filter, 0.0);
}
"""

sinh_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sinh(input);
}
"""

selu = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = selu(conv);
}
"""

prelu_4d_standalone = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external(shape = [16,16,32,32]);
    input2 = external(shape = [16]);
    output = prelu(input1, input2);
}
"""

tile_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = tile(input, repeats = [1,1,3,3]);
}
"""

softmax_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = softmax(input);
}
"""

rsqrt_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = rsqrt(input);
}
"""

concat_channel = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external(shape = [4,16,32,32]);
    input2 = external(shape = [4,16,32,32]);
    output = concat([input1, input2], axis = 1);
}
"""

area_downsample = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = area_downsample(input, factor = [2,2]);
}
"""

elu = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = elu(conv);
}
"""

max_pool3x3_pad1_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,1), (1,1)], border = 'ignore');
}
"""

sigmoid_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = sigmoid(input);
}
"""

ne_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = ne(input, 0.5);
}
"""

conv3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

all_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = all_reduce(input, axes = [1]);
}
"""

squeeze_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,1,1]);
    output = squeeze(input, axes = [2,3]);
}
"""

and_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = and(input, false);
}
"""

l1_normalization = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = l1_normalization(input, axes = [1], bias = 1.0, epsilon = 1e-5);
}
"""

max_pool3x3_constant_border = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'constant');
}
"""

argmax_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = argmax_reduce(input, axes = [2,3]);
}
"""

cos_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = cos(input);
}
"""

sqr_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sqr(input);
}
"""

rsqrt_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = rsqrt(input);
}
"""

acos_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = acos(input);
}
"""

bilinear_upsample_symmetric_replicate = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = multilinear_upsample(input, factor = [2,2], method = 'symmetric', border = 'replicate');
}
"""

asinh_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = asinh(input);
}
"""

tile_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,1]);
    output = tile(input, repeats = [1,16]);
}
"""

cosh_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = cosh(input);
}
"""

div_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = div(input1, input2);
}
"""

sqrt_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = sqrt(input);
}
"""

and_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<logical>(shape = [4,16,32,32]);
    input2 = external<logical>(shape = [1,16,1,1]);
    output = and(input1, input2);
}
"""

transpose_nhwc_to_nchw = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,32,32,16]);
    output = transpose(input, axes = [0,3,1,2]);
}
"""

avg_pool3x3_pad0_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,1), (0,1)], border = 'constant');
}
"""

round_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = round(input);
}
"""

box3x3_pad0_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,1), (0,1)], border = 'ignore');
}
"""

deconv6x6 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,6,6], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

atan_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = atan(input);
}
"""

add_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = add(input, 0.5);
}
"""

lt_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = lt(input1, input2);
}
"""

min_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = min(input1, input2);
}
"""

box3x3_stride1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,3,3], stride = [1,1,1,1], border = 'constant');
}
"""

linear_nobias = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    weights = variable(shape = [32,16], label = 'weights');
    output = linear(input, weights, 0.0);
}
"""

div_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = div(input1, input2);
}
"""

avg_pool3x3_stride1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,1,1], border = 'constant');
}
"""

conv7x7 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,7,7], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

conv3x3_groups0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,1,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, groups = 0);
}
"""

mul_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = mul(input1, input2);
}
"""

deconv3x3_pad1_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, padding = [(1,0), (1,0)]);
}
"""

ne_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = ne(input1, input2);
}
"""

tan_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = tan(input);
}
"""

avg_pool3x3_pad1_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,1), (1,1)], border = 'constant');
}
"""

mean_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = mean_reduce(input, axes = [1]);
}
"""

softplus_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = softplus(input);
}
"""

conv5x5 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,5,5], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

max_pool3x3_stride1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,1,1], border = 'ignore');
}
"""

pad_1_0_reflect = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (1,0), (1,0)], border = 'reflect');
}
"""

pad_1_0_replicate = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (1,0), (1,0)], border = 'replicate');
}
"""

separable_conv5x5 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    plane_filter = variable(shape = [8,1,5,5], label = 'plane_filter');
    point_filter = variable(shape = [16,8,1,1], label = 'point_filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = separable_conv(input, plane_filter, point_filter, bias);
}
"""

debox3x3_pad1_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,1), (1,1)], border = 'constant');
}
"""

avg_pool3x3_pad1_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,0), (1,0)], border = 'constant');
}
"""

bilinear_upsample_symmetric_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = multilinear_upsample(input, factor = [2,2], method = 'symmetric', border = 'constant');
}
"""

gt_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = gt(input1, input2);
}
"""

tanh_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = tanh(input);
}
"""

acosh_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = acosh(input);
}
"""

asin_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = asin(input);
}
"""

add_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = add(input1, input2);
}
"""

rsqr_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = rsqr(input);
}
"""

div_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = div(input1, input2);
}
"""

eq_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = eq(input1, input2);
}
"""

conv3x3_valid = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, padding = [(0,0), (0,0)]);
}
"""

min_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = min(input, 0.5);
}
"""

separable_deconv3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    plane_filter = variable(shape = [8,1,3,3], label = 'plane_filter');
    point_filter = variable(shape = [16,8,1,1], label = 'point_filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = separable_deconv(input, plane_filter, point_filter, bias);
}
"""

asin_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = asin(input);
}
"""

or_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = or(input, false);
}
"""

min_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = min_reduce(input, axes = [1]);
}
"""

silu = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = silu(conv);
}
"""

max_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = max_reduce(input, axes = [2,3]);
}
"""

bilinear_upsample_asymmetric_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = multilinear_upsample(input, factor = [2,2], method = 'asymmetric', border = 'constant');
}
"""

gelu = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = gelu(conv);
}
"""

clamp_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = clamp(input, 0.25, 0.75);
}
"""

conv3x3_pad0_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, padding = [(0,0), (0,0)]);
}
"""

conv3x3_pad1_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, padding = [(1,0), (1,0)]);
}
"""

abs_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = abs(input);
}
"""

max_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = max_reduce(input, axes = [1]);
}
"""

ge_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = ge(input, 0.5);
}
"""

pad_1_1_reflect = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (1,1), (1,1)], border = 'reflect');
}
"""

elu_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    output = elu(input);
}
"""

cosh_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = cosh(input);
}
"""

transpose_nchw_to_nhwc = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = transpose(input, axes = [0,2,3,1]);
}
"""

deconv3x3_pad1_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, padding = [(1,1), (1,1)]);
}
"""

ne_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = ne(input1, input2);
}
"""

sqr_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = sqr(input);
}
"""

conv3x3_pad1_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, padding = [(1,1), (1,1)]);
}
"""

clamp_4d = """
version 1.0;

graph G( input1, input2, input3 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    input3 = external<scalar>(shape = [4,16,32,32]);
    output = clamp(input1, input2, input3);
}
"""

bilinear_upsample_aligned_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = multilinear_upsample(input, factor = [2,2], method = 'aligned', border = 'constant');
}
"""

stack = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external(shape = [4,16,32,32]);
    input2 = external(shape = [4,16,32,32]);
    output = stack([input1, input2], axis = 1);
}
"""

log2_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = log2(input);
}
"""

slice = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = slice(input, axes = [2,3], begin = [1,2], end = [-1,-2]);
}
"""

deconv2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,2,2], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

all_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = all_reduce(input, axes = [2,3]);
}
"""

unstack = """
version 1.0;

graph G( input ) -> ( output1, output2, output3 )
{
    input = external(shape = [4,3,16]);
    [output1, output2, output3] = unstack(input, axis = 1);
}
"""

sqrt_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sqrt(input);
}
"""

l2_normalization = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = l2_normalization(input, axes = [1], epsilon = 1e-3);
}
"""

conv7x7_stride4x4 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,7,7], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, stride = [4,4]);
}
"""

ge_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = ge(input1, input2);
}
"""

any_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = any_reduce(input, axes = [1]);
}
"""

leaky_relu = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = leaky_relu(conv, alpha = 0.5);
}
"""

and_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<logical>(shape = [4,16]);
    input2 = external<logical>(shape = [4,16]);
    output = and(input1, input2);
}
"""

sinh_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = sinh(input);
}
"""

add_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = add(input1, input2);
}
"""

copy_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = copy(input);
}
"""

separable_conv3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    plane_filter = variable(shape = [8,1,3,3], label = 'plane_filter');
    point_filter = variable(shape = [16,8,1,1], label = 'point_filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = separable_conv(input, plane_filter, point_filter, bias);
}
"""

ceil_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = ceil(input);
}
"""

linear_squeeze = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,1,1]);
    weights = variable(shape = [32,16], label = 'weights');
    bias = variable(shape = [1,32], label = 'bias');
    squeezed = squeeze(input, axes = [2,3]);
    output = linear(squeezed, weights, bias);
}
"""

acosh_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = acosh(input);
}
"""

sub_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = sub(input1, input2);
}
"""

deconv3x3_valid = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, padding = [(0,0), (0,0)]);
}
"""

pow_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = pow(input1, input2);
}
"""

pad_1_1_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (1,1), (1,1)], border = 'constant');
}
"""

clamp_2d = """
version 1.0;

graph G( input1, input2, input3 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    input3 = external<scalar>(shape = [4,16]);
    output = clamp(input1, input2, input3);
}
"""

debox3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'constant');
}
"""

conv1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,1,1], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

exp_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = exp(input);
}
"""

avg_pool3x3_ignore_border = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'ignore');
}
"""

deconv3x3_pad0_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, padding = [(0,0), (0,0)]);
}
"""

leaky_relu_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    output = leaky_relu(input, alpha = 0.5);
}
"""

pow_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = pow(input1, input2);
}
"""

abs_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = abs(input);
}
"""

sin_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sin(input);
}
"""

select_2d_true = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = select(true, input1, input2);
}
"""

relu_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = relu(input);
}
"""

reshape_squeeze = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,1,1]);
    output = reshape(input, shape = [4,16]);
}
"""

selu_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16]);
    output = selu(input);
}
"""

sub_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = sub(input, 0.5);
}
"""

linear = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    weights = variable(shape = [32,16], label = 'weights');
    bias = variable(shape = [1,32], label = 'bias');
    output = linear(input, weights, bias);
}
"""

atanh_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = atanh(input);
}
"""

pow_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = pow(input1, input2);
}
"""

rms_pool3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = rms_pool(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'constant');
}
"""

debox3x3_pad0_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,1), (0,1)], border = 'ignore');
}
"""

floor_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = floor(input);
}
"""

deconv3x3_nobias = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    output = deconv(input, filter, 0.0);
}
"""

batch_norm = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    mean = variable(shape = [1,16], label = 'mean');
    variance = variable(shape = [1,16], label = 'variance');
    offset = variable(shape = [1,16], label = 'offset');
    scale = variable(shape = [1,16], label = 'scale');
    output = batch_normalization(input, mean, variance, offset, scale, epsilon = 1e-3);
}
"""

deconv3x3_stride2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, stride = [2,2]);
}
"""

debox2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,2,2], stride = [1,1,2,2], border = 'constant');
}
"""

gelu_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16]);
    output = gelu(input);
}
"""

pad_0_1_replicate = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (0,1), (0,1)], border = 'replicate');
}
"""

mul_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = mul(input1, input2);
}
"""

local_mean_normalization = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = local_mean_normalization(input, size = [1, 1, 3, 3]);
}
"""

debox3x3_pad0_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,0), (0,0)], border = 'constant');
}
"""

reshape_partial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [2,3,3,3,2]);
    output = reshape(input, shape = [0,-1], axis_start = 1, axis_count = 3);
}
"""

argmin_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = argmin_reduce(input, axes = [1]);
}
"""

softplus = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = softplus(conv);
}
"""

copy_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = copy(input);
}
"""

local_variance_normalization = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = local_variance_normalization(input, size = [1, 1, 3, 3], bias = 1.0, epsilon = 1e-5);
}
"""

not_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16]);
    output = not(input);
}
"""

sigmoid_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = sigmoid(input);
}
"""

local_response_normalization = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = local_response_normalization(input, alpha = 1e-05, beta = 0.75, bias = 1.0, size = [1, 5, 1, 1]);
}
"""

gelu_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16,32,32]);
    output = gelu(input);
}
"""

separable_conv3x3_with_attrs = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    plane_filter = variable(shape = [8,1,3,3], label = 'plane_filter');
    point_filter = variable(shape = [16,8,1,1], label = 'point_filter');
    output = separable_conv(input, plane_filter, point_filter, padding = [(0,1), (0,1)], stride = [2,2]);
}
"""

exp_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = exp(input);
}
"""

lt_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = lt(input1, input2);
}
"""

conv4x4 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,4,4], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

avg_pool3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'constant');
}
"""

avg_pool3x3_pad0_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = avg_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,0), (0,0)], border = 'constant');
}
"""

conv3x3_pad0_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, padding = [(0,1), (0,1)]);
}
"""

pad_0_1_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (0,1), (0,1)], border = 'constant');
}
"""

deconv4x4 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,4,4], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

neg_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = neg(input);
}
"""

bilinear_upsample_asymmetric_replicate = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = multilinear_upsample(input, factor = [2,2], method = 'asymmetric', border = 'replicate');
}
"""

conv5x5_stride3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,5,5], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, stride = [3,3]);
}
"""

relu_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = relu(input);
}
"""

max_pool1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,1,1], stride = [1,1,2,2], border = 'ignore');
}
"""

deconv5x5_pad2_2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,5,5], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, padding = [(2,2), (2,2)]);
}
"""

tile_batch = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16]);
    output = tile(input, repeats = [16,1]);
}
"""

eq_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = eq(input, 0.5);
}
"""

elu_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16]);
    output = elu(input);
}
"""

lt_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = lt(input1, input2);
}
"""

deconv1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,1,1], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias);
}
"""

sign_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = sign(input);
}
"""

leaky_relu_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16]);
    output = leaky_relu(input, alpha = 0.5);
}
"""

select_2d_false = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = select(false, input1, input2);
}
"""

div_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = div(input, 0.5);
}
"""

softplus_4d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = softplus(input);
}
"""

pow_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = pow(input, 0.5);
}
"""

round_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = round(input);
}
"""

debox3x3_stride1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = debox(input, size = [1,1,3,3], stride = [1,1,1,1], border = 'constant');
}
"""

separable_deconv3x3_with_attrs = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    plane_filter = variable(shape = [8,1,3,3], label = 'plane_filter');
    point_filter = variable(shape = [16,8,1,1], label = 'point_filter');
    output = separable_deconv(input, plane_filter, point_filter, padding = [(0,1), (0,1)], stride = [2,2]);
}
"""

matmul_2d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [16,4]);
    output = matmul(input1, input2);
}
"""

deconv5x5_stride3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,5,5], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, stride = [3,3]);
}
"""

sub_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = sub(input1, input2);
}
"""

matmul_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = matmul(input1, input2);
}
"""

any_reduce_spatial = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<logical>(shape = [4,16,32,32]);
    output = any_reduce(input, axes = [2,3]);
}
"""

gt_4d_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = gt(input, 0.5);
}
"""

conv6x6 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,6,6], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias);
}
"""

le_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = le(input1, input2);
}
"""

gt_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = gt(input1, input2);
}
"""

deconv4x4_stride2x2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,4,4], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, stride = [2,2]);
}
"""

le_4d_broadcast = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [1,16,1,1]);
    output = le(input1, input2);
}
"""

tanh_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = tanh(input);
}
"""

split_unbalanced = """
version 1.0;

graph G( input ) -> ( output1, output2, output3 )
{
    input = external(shape = [4,32,3]);
    [output1, output2, output3] = split(input, axis = 1, ratios = [3,1,4]);
}
"""

box3x3 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,3,3], stride = [1,1,2,2], border = 'constant');
}
"""

select_4d_false = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = select(false, input1, input2);
}
"""

tanh = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = tanh(conv);
}
"""

sin_2d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16]);
    output = sin(input);
}
"""

box3x3_pad0_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,0), (0,0)], border = 'constant');
}
"""

box1x1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,1,1], stride = [1,1,2,2], border = 'constant');
}
"""

box3x3_pad1_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = box(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (1,1), (1,1)], border = 'constant');
}
"""

conv5x5_pad2_2 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,8,32,32]);
    filter = variable(shape = [16,8,5,5], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = conv(input, filter, bias, padding = [(2,2), (2,2)]);
}
"""

prelu_2d_standalone = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external(shape = [16,16]);
    input2 = external(shape = [16]);
    output = prelu(input1, input2);
}
"""

max_pool3x3_pad0_0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    output = max_pool(input, size = [1,1,3,3], stride = [1,1,2,2], padding = [(0,0), (0,0), (0,0), (0,0)], border = 'ignore');
}
"""

softmax_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = softmax(input);
}
"""

matmul_2d_transpose = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16]);
    input2 = external<scalar>(shape = [4,16]);
    output = matmul(input1, input2, transposeA = true, transposeB = false);
}
"""

silu_2d_standalone = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [16,16]);
    output = silu(input);
}
"""

deconv3x3_groups0 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,1,3,3], label = 'filter');
    bias = variable(shape = [1,16], label = 'bias');
    output = deconv(input, filter, bias, groups = 0);
}
"""

deconv3x3_pad0_1 = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = variable(shape = [16,8,3,3], label = 'filter');
    bias = variable(shape = [1,8], label = 'bias');
    output = deconv(input, filter, bias, padding = [(0,1), (0,1)]);
}
"""

sigmoid = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16,32,32]);
    filter = constant(shape = [16,1,1,1], value = [1.0]);
    bias = constant(shape = [1,16], value = [0.0]);
    conv = conv(input, filter, bias, groups = 0);
    output = sigmoid(conv);
}
"""

argmax_reduce_channel = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = argmax_reduce(input, axes = [1]);
}
"""

pad_1_1_replicate = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (1,1), (1,1)], border = 'replicate');
}
"""

pad_1_0_constant = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [1,16,32,32]);
    output = pad(input, padding = [(0,0), (0,0), (1,0), (1,0)], border = 'constant');
}
"""

unsqueeze = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external(shape = [4,16]);
    output = unsqueeze(input, axes = [2,3]);
}
"""

neg_4d = """
version 1.0;

graph G( input ) -> ( output )
{
    input = external<scalar>(shape = [4,16,32,32]);
    output = neg(input);
}
"""

add_4d = """
version 1.0;

graph G( input1, input2 ) -> ( output )
{
    input1 = external<scalar>(shape = [4,16,32,32]);
    input2 = external<scalar>(shape = [4,16,32,32]);
    output = add(input1, input2);
}
"""
