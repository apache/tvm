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

import pytest
import functools
import numpy as np

# jax packages
import jax
from jax import random
from tests.python.relax.jax_resnet import ResNet18

# mlir and stablehlo from jaxlib
from jaxlib.mlir.ir import *
from jaxlib.mlir.dialects import stablehlo

import tvm
from tvm import relax, tir
from tvm.script.parser import relax as R
from tvm.relax.frontend.stablehlo import from_stablehlo


def to_relax(model: str) -> tvm.ir.IRModule:
    with Context() as context:
        stablehlo.register_dialect(context)
        m = Module.parse(model)
    ir_mod = from_stablehlo(m)
    # ir_mod.show()
    return ir_mod


def test_add_dynamic():
    ASM = """
    func.func @test(%arg0: tensor<3x?xf32>, %arg1: tensor<3x?xf32>) -> tensor<3x?xf32> {
      %1 = stablehlo.add %arg0, %arg1 : (tensor<3x?xf32>, tensor<3x?xf32>) -> tensor<3x?xf32>
      func.return %1 : tensor<3x?xf32>
    }
    """
    to_relax(ASM)


def test_add():
    ASM = """
    module @jit_f {
      func.func public @main(%arg0: tensor<2x3xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x3xf32> {jax.arg_info = "y", mhlo.sharding = "{replicated}"}) -> (tensor<2x3xf32> {jax.result_info = ""}) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
        %1 = stablehlo.add %0, %arg1 : tensor<2x3xf32>
        return %1 : tensor<2x3xf32>
      }
    }
    """
    to_relax(ASM)


def test_resnet():

    model = ResNet18(num_classes=2, dtype=np.float32)
    x = np.zeros((8, 16, 16, 3), np.float32)
    variables = model.init(random.PRNGKey(0), x)
    apply = functools.partial(model.apply, train=False, mutable=False)
    key = jax.random.PRNGKey(0)
    input_shape = (1, 224, 224, 3)
    x = jax.random.normal(key, input_shape)
    resnet50_jit = jax.jit(apply)
    y = resnet50_jit(variables, x)
    print("y is :", y)
    lowered = jax.jit(apply).lower(variables, x)
    stablehlo_ir = lowered.as_text(dialect="stablehlo")
    print(stablehlo_ir)


def test_resnet18():
    ASM = """
module @jit__unnamed_wrapped_function_ {
  func.func public @main(%arg0: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_0']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg1: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_0']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg2: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_0']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg3: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_0']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg4: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_1']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg5: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_1']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg6: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_1']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg7: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_1']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg8: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_2']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg9: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_2']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg10: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_2']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg11: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_2']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg12: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_2']['norm_proj']['mean']", mhlo.sharding = "{replicated}"}, %arg13: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_2']['norm_proj']['var']", mhlo.sharding = "{replicated}"}, %arg14: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_3']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg15: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_3']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg16: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_3']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg17: tensor<128xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_3']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg18: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_4']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg19: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_4']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg20: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_4']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg21: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_4']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg22: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_4']['norm_proj']['mean']", mhlo.sharding = "{replicated}"}, %arg23: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_4']['norm_proj']['var']", mhlo.sharding = "{replicated}"}, %arg24: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_5']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg25: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_5']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg26: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_5']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg27: tensor<256xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_5']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg28: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_6']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg29: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_6']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg30: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_6']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg31: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_6']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg32: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_6']['norm_proj']['mean']", mhlo.sharding = "{replicated}"}, %arg33: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_6']['norm_proj']['var']", mhlo.sharding = "{replicated}"}, %arg34: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_7']['BatchNorm_0']['mean']", mhlo.sharding = "{replicated}"}, %arg35: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_7']['BatchNorm_0']['var']", mhlo.sharding = "{replicated}"}, %arg36: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_7']['BatchNorm_1']['mean']", mhlo.sharding = "{replicated}"}, %arg37: tensor<512xf32> {jax.arg_info = "variables['batch_stats']['ResNetBlock_7']['BatchNorm_1']['var']", mhlo.sharding = "{replicated}"}, %arg38: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['bn_init']['mean']", mhlo.sharding = "{replicated}"}, %arg39: tensor<64xf32> {jax.arg_info = "variables['batch_stats']['bn_init']['var']", mhlo.sharding = "{replicated}"}, %arg40: tensor<2xf32> {jax.arg_info = "variables['params']['Dense_0']['bias']", mhlo.sharding = "{replicated}"}, %arg41: tensor<512x2xf32> {jax.arg_info = "variables['params']['Dense_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg42: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_0']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg43: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_0']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg44: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_0']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg45: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_0']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg46: tensor<3x3x64x64xf32> {jax.arg_info = "variables['params']['ResNetBlock_0']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg47: tensor<3x3x64x64xf32> {jax.arg_info = "variables['params']['ResNetBlock_0']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg48: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_1']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg49: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_1']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg50: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_1']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg51: tensor<64xf32> {jax.arg_info = "variables['params']['ResNetBlock_1']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg52: tensor<3x3x64x64xf32> {jax.arg_info = "variables['params']['ResNetBlock_1']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg53: tensor<3x3x64x64xf32> {jax.arg_info = "variables['params']['ResNetBlock_1']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg54: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg55: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg56: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg57: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg58: tensor<3x3x64x128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg59: tensor<3x3x128x128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg60: tensor<1x1x64x128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['conv_proj']['kernel']", mhlo.sharding = "{replicated}"}, %arg61: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['norm_proj']['bias']", mhlo.sharding = "{replicated}"}, %arg62: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_2']['norm_proj']['scale']", mhlo.sharding = "{replicated}"}, %arg63: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_3']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg64: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_3']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg65: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_3']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg66: tensor<128xf32> {jax.arg_info = "variables['params']['ResNetBlock_3']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg67: tensor<3x3x128x128xf32> {jax.arg_info = "variables['params']['ResNetBlock_3']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg68: tensor<3x3x128x128xf32> {jax.arg_info = "variables['params']['ResNetBlock_3']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg69: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg70: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg71: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg72: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg73: tensor<3x3x128x256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg74: tensor<3x3x256x256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg75: tensor<1x1x128x256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['conv_proj']['kernel']", mhlo.sharding = "{replicated}"}, %arg76: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['norm_proj']['bias']", mhlo.sharding = "{replicated}"}, %arg77: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_4']['norm_proj']['scale']", mhlo.sharding = "{replicated}"}, %arg78: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_5']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg79: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_5']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg80: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_5']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg81: tensor<256xf32> {jax.arg_info = "variables['params']['ResNetBlock_5']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg82: tensor<3x3x256x256xf32> {jax.arg_info = "variables['params']['ResNetBlock_5']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg83: tensor<3x3x256x256xf32> {jax.arg_info = "variables['params']['ResNetBlock_5']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg84: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg85: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg86: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg87: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg88: tensor<3x3x256x512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg89: tensor<3x3x512x512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg90: tensor<1x1x256x512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['conv_proj']['kernel']", mhlo.sharding = "{replicated}"}, %arg91: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['norm_proj']['bias']", mhlo.sharding = "{replicated}"}, %arg92: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_6']['norm_proj']['scale']", mhlo.sharding = "{replicated}"}, %arg93: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_7']['BatchNorm_0']['bias']", mhlo.sharding = "{replicated}"}, %arg94: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_7']['BatchNorm_0']['scale']", mhlo.sharding = "{replicated}"}, %arg95: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_7']['BatchNorm_1']['bias']", mhlo.sharding = "{replicated}"}, %arg96: tensor<512xf32> {jax.arg_info = "variables['params']['ResNetBlock_7']['BatchNorm_1']['scale']", mhlo.sharding = "{replicated}"}, %arg97: tensor<3x3x512x512xf32> {jax.arg_info = "variables['params']['ResNetBlock_7']['Conv_0']['kernel']", mhlo.sharding = "{replicated}"}, %arg98: tensor<3x3x512x512xf32> {jax.arg_info = "variables['params']['ResNetBlock_7']['Conv_1']['kernel']", mhlo.sharding = "{replicated}"}, %arg99: tensor<64xf32> {jax.arg_info = "variables['params']['bn_init']['bias']", mhlo.sharding = "{replicated}"}, %arg100: tensor<64xf32> {jax.arg_info = "variables['params']['bn_init']['scale']", mhlo.sharding = "{replicated}"}, %arg101: tensor<7x7x3x64xf32> {jax.arg_info = "variables['params']['conv_init']['kernel']", mhlo.sharding = "{replicated}"}, %arg102: tensor<1x224x224x3xf32> {jax.arg_info = "args[0]", mhlo.sharding = "{replicated}"}) -> (tensor<1x2xf32> {jax.result_info = ""}) {
    %0 = stablehlo.convolution(%arg102, %arg101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
    %1 = stablehlo.reshape %arg38 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2 = stablehlo.reshape %arg39 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<1x112x112x64xf32>
    %5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x1x1x64xf32>
    %8 = stablehlo.rsqrt %7 : tensor<1x1x1x64xf32>
    %9 = stablehlo.reshape %arg100 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1x112x112x64xf32>
    %13 = stablehlo.reshape %arg99 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x112x112x64xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<1x112x112x64xf32>
    %18 = stablehlo.maximum %15, %17 : tensor<1x112x112x64xf32>
    %19 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<f32>
    %21 = "stablehlo.reduce_window"(%18, %20) ({
    ^bb0(%arg103: tensor<f32>, %arg104: tensor<f32>):
      %390 = stablehlo.maximum %arg103, %arg104 : tensor<f32>
      stablehlo.return %390 : tensor<f32>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
    %22 = stablehlo.convolution(%21, %arg46) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %23 = stablehlo.reshape %arg0 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %24 = stablehlo.reshape %arg1 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %25 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %26 = stablehlo.subtract %22, %25 : tensor<1x56x56x64xf32>
    %27 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %29 = stablehlo.add %24, %28 : tensor<1x1x1x64xf32>
    %30 = stablehlo.rsqrt %29 : tensor<1x1x1x64xf32>
    %31 = stablehlo.reshape %arg43 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<1x1x1x64xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %34 = stablehlo.multiply %26, %33 : tensor<1x56x56x64xf32>
    %35 = stablehlo.reshape %arg42 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %37 = stablehlo.add %34, %36 : tensor<1x56x56x64xf32>
    %38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %40 = stablehlo.maximum %37, %39 : tensor<1x56x56x64xf32>
    %41 = stablehlo.convolution(%40, %arg47) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %42 = stablehlo.reshape %arg2 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %43 = stablehlo.reshape %arg3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %45 = stablehlo.subtract %41, %44 : tensor<1x56x56x64xf32>
    %46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %48 = stablehlo.add %43, %47 : tensor<1x1x1x64xf32>
    %49 = stablehlo.rsqrt %48 : tensor<1x1x1x64xf32>
    %50 = stablehlo.reshape %arg45 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %51 = stablehlo.multiply %49, %50 : tensor<1x1x1x64xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %53 = stablehlo.multiply %45, %52 : tensor<1x56x56x64xf32>
    %54 = stablehlo.reshape %arg44 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %56 = stablehlo.add %53, %55 : tensor<1x56x56x64xf32>
    %57 = stablehlo.add %21, %56 : tensor<1x56x56x64xf32>
    %58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %60 = stablehlo.maximum %57, %59 : tensor<1x56x56x64xf32>
    %61 = stablehlo.convolution(%60, %arg52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %62 = stablehlo.reshape %arg4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %63 = stablehlo.reshape %arg5 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %64 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %65 = stablehlo.subtract %61, %64 : tensor<1x56x56x64xf32>
    %66 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %68 = stablehlo.add %63, %67 : tensor<1x1x1x64xf32>
    %69 = stablehlo.rsqrt %68 : tensor<1x1x1x64xf32>
    %70 = stablehlo.reshape %arg49 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %71 = stablehlo.multiply %69, %70 : tensor<1x1x1x64xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %73 = stablehlo.multiply %65, %72 : tensor<1x56x56x64xf32>
    %74 = stablehlo.reshape %arg48 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %76 = stablehlo.add %73, %75 : tensor<1x56x56x64xf32>
    %77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %79 = stablehlo.maximum %76, %78 : tensor<1x56x56x64xf32>
    %80 = stablehlo.convolution(%79, %arg53) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %81 = stablehlo.reshape %arg6 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %82 = stablehlo.reshape %arg7 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %83 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %84 = stablehlo.subtract %80, %83 : tensor<1x56x56x64xf32>
    %85 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %87 = stablehlo.add %82, %86 : tensor<1x1x1x64xf32>
    %88 = stablehlo.rsqrt %87 : tensor<1x1x1x64xf32>
    %89 = stablehlo.reshape %arg51 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %90 = stablehlo.multiply %88, %89 : tensor<1x1x1x64xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %92 = stablehlo.multiply %84, %91 : tensor<1x56x56x64xf32>
    %93 = stablehlo.reshape %arg50 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %95 = stablehlo.add %92, %94 : tensor<1x56x56x64xf32>
    %96 = stablehlo.add %60, %95 : tensor<1x56x56x64xf32>
    %97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %99 = stablehlo.maximum %96, %98 : tensor<1x56x56x64xf32>
    %100 = stablehlo.convolution(%99, %arg58) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x28x28x128xf32>
    %101 = stablehlo.reshape %arg8 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %102 = stablehlo.reshape %arg9 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %103 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %104 = stablehlo.subtract %100, %103 : tensor<1x28x28x128xf32>
    %105 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %107 = stablehlo.add %102, %106 : tensor<1x1x1x128xf32>
    %108 = stablehlo.rsqrt %107 : tensor<1x1x1x128xf32>
    %109 = stablehlo.reshape %arg55 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<1x1x1x128xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %112 = stablehlo.multiply %104, %111 : tensor<1x28x28x128xf32>
    %113 = stablehlo.reshape %arg54 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %115 = stablehlo.add %112, %114 : tensor<1x28x28x128xf32>
    %116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [] : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %118 = stablehlo.maximum %115, %117 : tensor<1x28x28x128xf32>
    %119 = stablehlo.convolution(%118, %arg59) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %120 = stablehlo.reshape %arg10 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %121 = stablehlo.reshape %arg11 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %122 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %123 = stablehlo.subtract %119, %122 : tensor<1x28x28x128xf32>
    %124 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %125 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %126 = stablehlo.add %121, %125 : tensor<1x1x1x128xf32>
    %127 = stablehlo.rsqrt %126 : tensor<1x1x1x128xf32>
    %128 = stablehlo.reshape %arg57 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %129 = stablehlo.multiply %127, %128 : tensor<1x1x1x128xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %131 = stablehlo.multiply %123, %130 : tensor<1x28x28x128xf32>
    %132 = stablehlo.reshape %arg56 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %134 = stablehlo.add %131, %133 : tensor<1x28x28x128xf32>
    %135 = stablehlo.convolution(%99, %arg60) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<1x1x64x128xf32>) -> tensor<1x28x28x128xf32>
    %136 = stablehlo.reshape %arg12 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %137 = stablehlo.reshape %arg13 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %138 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %139 = stablehlo.subtract %135, %138 : tensor<1x28x28x128xf32>
    %140 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %141 = stablehlo.broadcast_in_dim %140, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %142 = stablehlo.add %137, %141 : tensor<1x1x1x128xf32>
    %143 = stablehlo.rsqrt %142 : tensor<1x1x1x128xf32>
    %144 = stablehlo.reshape %arg62 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %145 = stablehlo.multiply %143, %144 : tensor<1x1x1x128xf32>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %147 = stablehlo.multiply %139, %146 : tensor<1x28x28x128xf32>
    %148 = stablehlo.reshape %arg61 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %150 = stablehlo.add %147, %149 : tensor<1x28x28x128xf32>
    %151 = stablehlo.add %150, %134 : tensor<1x28x28x128xf32>
    %152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %154 = stablehlo.maximum %151, %153 : tensor<1x28x28x128xf32>
    %155 = stablehlo.convolution(%154, %arg67) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %156 = stablehlo.reshape %arg14 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %157 = stablehlo.reshape %arg15 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %158 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %159 = stablehlo.subtract %155, %158 : tensor<1x28x28x128xf32>
    %160 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %162 = stablehlo.add %157, %161 : tensor<1x1x1x128xf32>
    %163 = stablehlo.rsqrt %162 : tensor<1x1x1x128xf32>
    %164 = stablehlo.reshape %arg64 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %165 = stablehlo.multiply %163, %164 : tensor<1x1x1x128xf32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %167 = stablehlo.multiply %159, %166 : tensor<1x28x28x128xf32>
    %168 = stablehlo.reshape %arg63 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %170 = stablehlo.add %167, %169 : tensor<1x28x28x128xf32>
    %171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %173 = stablehlo.maximum %170, %172 : tensor<1x28x28x128xf32>
    %174 = stablehlo.convolution(%173, %arg68) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %175 = stablehlo.reshape %arg16 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %176 = stablehlo.reshape %arg17 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %177 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %178 = stablehlo.subtract %174, %177 : tensor<1x28x28x128xf32>
    %179 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %181 = stablehlo.add %176, %180 : tensor<1x1x1x128xf32>
    %182 = stablehlo.rsqrt %181 : tensor<1x1x1x128xf32>
    %183 = stablehlo.reshape %arg66 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %184 = stablehlo.multiply %182, %183 : tensor<1x1x1x128xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %186 = stablehlo.multiply %178, %185 : tensor<1x28x28x128xf32>
    %187 = stablehlo.reshape %arg65 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %189 = stablehlo.add %186, %188 : tensor<1x28x28x128xf32>
    %190 = stablehlo.add %154, %189 : tensor<1x28x28x128xf32>
    %191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [] : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %193 = stablehlo.maximum %190, %192 : tensor<1x28x28x128xf32>
    %194 = stablehlo.convolution(%193, %arg73) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x128xf32>, tensor<3x3x128x256xf32>) -> tensor<1x14x14x256xf32>
    %195 = stablehlo.reshape %arg18 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %196 = stablehlo.reshape %arg19 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %197 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %198 = stablehlo.subtract %194, %197 : tensor<1x14x14x256xf32>
    %199 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %201 = stablehlo.add %196, %200 : tensor<1x1x1x256xf32>
    %202 = stablehlo.rsqrt %201 : tensor<1x1x1x256xf32>
    %203 = stablehlo.reshape %arg70 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %204 = stablehlo.multiply %202, %203 : tensor<1x1x1x256xf32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %206 = stablehlo.multiply %198, %205 : tensor<1x14x14x256xf32>
    %207 = stablehlo.reshape %arg69 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %209 = stablehlo.add %206, %208 : tensor<1x14x14x256xf32>
    %210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [] : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %212 = stablehlo.maximum %209, %211 : tensor<1x14x14x256xf32>
    %213 = stablehlo.convolution(%212, %arg74) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %214 = stablehlo.reshape %arg20 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %215 = stablehlo.reshape %arg21 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %216 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %217 = stablehlo.subtract %213, %216 : tensor<1x14x14x256xf32>
    %218 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %219 = stablehlo.broadcast_in_dim %218, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %220 = stablehlo.add %215, %219 : tensor<1x1x1x256xf32>
    %221 = stablehlo.rsqrt %220 : tensor<1x1x1x256xf32>
    %222 = stablehlo.reshape %arg72 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %223 = stablehlo.multiply %221, %222 : tensor<1x1x1x256xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %225 = stablehlo.multiply %217, %224 : tensor<1x14x14x256xf32>
    %226 = stablehlo.reshape %arg71 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %228 = stablehlo.add %225, %227 : tensor<1x14x14x256xf32>
    %229 = stablehlo.convolution(%193, %arg75) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x128xf32>, tensor<1x1x128x256xf32>) -> tensor<1x14x14x256xf32>
    %230 = stablehlo.reshape %arg22 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %231 = stablehlo.reshape %arg23 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %232 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %233 = stablehlo.subtract %229, %232 : tensor<1x14x14x256xf32>
    %234 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %236 = stablehlo.add %231, %235 : tensor<1x1x1x256xf32>
    %237 = stablehlo.rsqrt %236 : tensor<1x1x1x256xf32>
    %238 = stablehlo.reshape %arg77 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %239 = stablehlo.multiply %237, %238 : tensor<1x1x1x256xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %241 = stablehlo.multiply %233, %240 : tensor<1x14x14x256xf32>
    %242 = stablehlo.reshape %arg76 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %244 = stablehlo.add %241, %243 : tensor<1x14x14x256xf32>
    %245 = stablehlo.add %244, %228 : tensor<1x14x14x256xf32>
    %246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %247 = stablehlo.broadcast_in_dim %246, dims = [] : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %248 = stablehlo.maximum %245, %247 : tensor<1x14x14x256xf32>
    %249 = stablehlo.convolution(%248, %arg82) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %250 = stablehlo.reshape %arg24 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %251 = stablehlo.reshape %arg25 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %252 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %253 = stablehlo.subtract %249, %252 : tensor<1x14x14x256xf32>
    %254 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %256 = stablehlo.add %251, %255 : tensor<1x1x1x256xf32>
    %257 = stablehlo.rsqrt %256 : tensor<1x1x1x256xf32>
    %258 = stablehlo.reshape %arg79 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %259 = stablehlo.multiply %257, %258 : tensor<1x1x1x256xf32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %261 = stablehlo.multiply %253, %260 : tensor<1x14x14x256xf32>
    %262 = stablehlo.reshape %arg78 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %263 = stablehlo.broadcast_in_dim %262, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %264 = stablehlo.add %261, %263 : tensor<1x14x14x256xf32>
    %265 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %266 = stablehlo.broadcast_in_dim %265, dims = [] : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %267 = stablehlo.maximum %264, %266 : tensor<1x14x14x256xf32>
    %268 = stablehlo.convolution(%267, %arg83) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %269 = stablehlo.reshape %arg26 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %270 = stablehlo.reshape %arg27 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %271 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %272 = stablehlo.subtract %268, %271 : tensor<1x14x14x256xf32>
    %273 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %275 = stablehlo.add %270, %274 : tensor<1x1x1x256xf32>
    %276 = stablehlo.rsqrt %275 : tensor<1x1x1x256xf32>
    %277 = stablehlo.reshape %arg81 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %278 = stablehlo.multiply %276, %277 : tensor<1x1x1x256xf32>
    %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %280 = stablehlo.multiply %272, %279 : tensor<1x14x14x256xf32>
    %281 = stablehlo.reshape %arg80 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %283 = stablehlo.add %280, %282 : tensor<1x14x14x256xf32>
    %284 = stablehlo.add %248, %283 : tensor<1x14x14x256xf32>
    %285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [] : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %287 = stablehlo.maximum %284, %286 : tensor<1x14x14x256xf32>
    %288 = stablehlo.convolution(%287, %arg88) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x256xf32>, tensor<3x3x256x512xf32>) -> tensor<1x7x7x512xf32>
    %289 = stablehlo.reshape %arg28 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %290 = stablehlo.reshape %arg29 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %291 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %292 = stablehlo.subtract %288, %291 : tensor<1x7x7x512xf32>
    %293 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %294 = stablehlo.broadcast_in_dim %293, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %295 = stablehlo.add %290, %294 : tensor<1x1x1x512xf32>
    %296 = stablehlo.rsqrt %295 : tensor<1x1x1x512xf32>
    %297 = stablehlo.reshape %arg85 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %298 = stablehlo.multiply %296, %297 : tensor<1x1x1x512xf32>
    %299 = stablehlo.broadcast_in_dim %298, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %300 = stablehlo.multiply %292, %299 : tensor<1x7x7x512xf32>
    %301 = stablehlo.reshape %arg84 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %303 = stablehlo.add %300, %302 : tensor<1x7x7x512xf32>
    %304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %305 = stablehlo.broadcast_in_dim %304, dims = [] : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %306 = stablehlo.maximum %303, %305 : tensor<1x7x7x512xf32>
    %307 = stablehlo.convolution(%306, %arg89) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %308 = stablehlo.reshape %arg30 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %309 = stablehlo.reshape %arg31 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %310 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %311 = stablehlo.subtract %307, %310 : tensor<1x7x7x512xf32>
    %312 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %313 = stablehlo.broadcast_in_dim %312, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %314 = stablehlo.add %309, %313 : tensor<1x1x1x512xf32>
    %315 = stablehlo.rsqrt %314 : tensor<1x1x1x512xf32>
    %316 = stablehlo.reshape %arg87 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %317 = stablehlo.multiply %315, %316 : tensor<1x1x1x512xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %319 = stablehlo.multiply %311, %318 : tensor<1x7x7x512xf32>
    %320 = stablehlo.reshape %arg86 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %322 = stablehlo.add %319, %321 : tensor<1x7x7x512xf32>
    %323 = stablehlo.convolution(%287, %arg90) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x256xf32>, tensor<1x1x256x512xf32>) -> tensor<1x7x7x512xf32>
    %324 = stablehlo.reshape %arg32 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %325 = stablehlo.reshape %arg33 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %326 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %327 = stablehlo.subtract %323, %326 : tensor<1x7x7x512xf32>
    %328 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %330 = stablehlo.add %325, %329 : tensor<1x1x1x512xf32>
    %331 = stablehlo.rsqrt %330 : tensor<1x1x1x512xf32>
    %332 = stablehlo.reshape %arg92 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %333 = stablehlo.multiply %331, %332 : tensor<1x1x1x512xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %335 = stablehlo.multiply %327, %334 : tensor<1x7x7x512xf32>
    %336 = stablehlo.reshape %arg91 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %338 = stablehlo.add %335, %337 : tensor<1x7x7x512xf32>
    %339 = stablehlo.add %338, %322 : tensor<1x7x7x512xf32>
    %340 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [] : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %342 = stablehlo.maximum %339, %341 : tensor<1x7x7x512xf32>
    %343 = stablehlo.convolution(%342, %arg97) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %344 = stablehlo.reshape %arg34 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %345 = stablehlo.reshape %arg35 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %346 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %347 = stablehlo.subtract %343, %346 : tensor<1x7x7x512xf32>
    %348 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %349 = stablehlo.broadcast_in_dim %348, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %350 = stablehlo.add %345, %349 : tensor<1x1x1x512xf32>
    %351 = stablehlo.rsqrt %350 : tensor<1x1x1x512xf32>
    %352 = stablehlo.reshape %arg94 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %353 = stablehlo.multiply %351, %352 : tensor<1x1x1x512xf32>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %355 = stablehlo.multiply %347, %354 : tensor<1x7x7x512xf32>
    %356 = stablehlo.reshape %arg93 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %358 = stablehlo.add %355, %357 : tensor<1x7x7x512xf32>
    %359 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %360 = stablehlo.broadcast_in_dim %359, dims = [] : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %361 = stablehlo.maximum %358, %360 : tensor<1x7x7x512xf32>
    %362 = stablehlo.convolution(%361, %arg98) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %363 = stablehlo.reshape %arg36 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %364 = stablehlo.reshape %arg37 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %365 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %366 = stablehlo.subtract %362, %365 : tensor<1x7x7x512xf32>
    %367 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %369 = stablehlo.add %364, %368 : tensor<1x1x1x512xf32>
    %370 = stablehlo.rsqrt %369 : tensor<1x1x1x512xf32>
    %371 = stablehlo.reshape %arg96 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %372 = stablehlo.multiply %370, %371 : tensor<1x1x1x512xf32>
    %373 = stablehlo.broadcast_in_dim %372, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %374 = stablehlo.multiply %366, %373 : tensor<1x7x7x512xf32>
    %375 = stablehlo.reshape %arg95 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %377 = stablehlo.add %374, %376 : tensor<1x7x7x512xf32>
    %378 = stablehlo.add %342, %377 : tensor<1x7x7x512xf32>
    %379 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [] : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %381 = stablehlo.maximum %378, %380 : tensor<1x7x7x512xf32>
    %382 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %383 = stablehlo.reduce(%381 init: %382) across dimensions = [1, 2] : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x512xf32>
     reducer(%arg103: tensor<f32>, %arg104: tensor<f32>)  {
      %390 = stablehlo.add %arg103, %arg104 : tensor<f32>
      stablehlo.return %390 : tensor<f32>
    }
    %384 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [] : (tensor<f32>) -> tensor<1x512xf32>
    %386 = stablehlo.divide %383, %385 : tensor<1x512xf32>
    %387 = "stablehlo.dot_general"(%386, %arg41) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x512xf32>, tensor<512x2xf32>) -> tensor<1x2xf32>
    %388 = stablehlo.reshape %arg40 : (tensor<2xf32>) -> tensor<1x2xf32>
    %389 = stablehlo.add %387, %388 : tensor<1x2xf32>
    return %389 : tensor<1x2xf32>
  }
}
    """

    mod = to_relax(ASM)
    print(mod)
    tvm_model = relax.transform.LegalizeOps()(mod)
    # tvm_model.show()

    jax_resnet18 = ResNet18(num_classes=2, dtype=np.float32)
    x = np.zeros((8, 16, 16, 3), np.float32)
    variables = jax_resnet18.init(random.PRNGKey(0), x)
    apply = functools.partial(jax_resnet18.apply, train=False, mutable=False)
    # print(apply)
    key = jax.random.PRNGKey(0)
    input_shape = (1, 224, 224, 3)
    x = jax.random.normal(key, input_shape)

    resnet50_jit = jax.jit(apply)
    print(resnet50_jit)
    print("variables keys: ", variables.keys())
    y = resnet50_jit(variables, x)
    print("y is :", y)


if __name__ == "__main__":
    # test_add()
    # test_add_dynamic()
    # test_resnet()
    test_resnet18()
