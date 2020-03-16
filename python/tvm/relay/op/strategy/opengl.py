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
"""Definition of OpenGL operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import topi
from .generic import *
from .. import op as _op

@schedule_injective.register("opengl")
def schedule_injective_opengl(attrs, outs, target):
    """schedule injective ops for opengl"""
    with target:
        return topi.opengl.schedule_injective(outs)

@schedule_concatenate.register("opengl")
def schedule_concatenate_opengl(attrs, outs, target):
    """schedule concatenate for opengl"""
    with target:
        return topi.opengl.schedule_injective(outs)

@schedule_pool.register("opengl")
def schedule_pool_opengl(attrs, outs, target):
    """schedule pooling ops for opengl"""
    with target:
        return topi.opengl.schedule_pool(outs, attrs.layout)

@schedule_adaptive_pool.register("opengl")
def schedule_adaptive_pool_opengl(attrs, outs, target):
    """schedule adative pooling ops for opengl"""
    with target:
        return topi.opengl.schedule_adaptive_pool(outs)

@schedule_softmax.register("opengl")
def schedule_softmax_opengl(attrs, outs, target):
    """schedule softmax for opengl"""
    with target:
        return topi.opengl.schedule_softmax(outs)

@conv2d_strategy.register("opengl")
def conv2d_strategy_opengl(attrs, inputs, out_type, target):
    """conv2d opengl strategy"""
    strategy = _op.OpStrategy()
    groups = attrs.groups
    layout = attrs.data_layout
    assert groups == 1, "Don't support group conv2d on OpenGL"
    assert layout == "NCHW", "Only support conv2d layout NCHW for OpenGL"
    strategy.add_implementation(wrap_compute_conv2d(topi.nn.conv2d),
                                wrap_topi_schedule(topi.opengl.schedule_conv2d_nchw),
                                name="conv2d_nchw.opengl")
    return strategy

@dense_strategy.register("opengl")
def dense_strategy_opengl(attrs, inputs, out_type, target):
    """dense opengl strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(wrap_compute_dense(topi.nn.dense),
                                wrap_topi_schedule(topi.opengl.schedule_dense),
                                name="dense.opengl")
    return strategy
