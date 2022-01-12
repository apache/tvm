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
"""Test DNNL integration conv2d tests."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.dnnl import partition_for_dnnl

from common import requires_dnnl, parametrized, check_result, Builder

import collections

BinaryShapeConfig = collections.namedtuple("BinaryShapeConfig", ["lhs_shape", "rhs_shape"])
base_0D = BinaryShapeConfig(lhs_shape=[], rhs_shape=[])
base_3D = BinaryShapeConfig(lhs_shape=[3, 2, 1], rhs_shape=[3, 2, 1])
base_4D = BinaryShapeConfig(lhs_shape=[4, 3, 2, 1], rhs_shape=[4, 3, 2, 1])
base_6D = BinaryShapeConfig(lhs_shape=[2, 3, 4, 3, 2, 1], rhs_shape=[2, 3, 4, 3, 2, 1])

scalar_broadcast_6D = BinaryShapeConfig(lhs_shape=[2, 3, 4, 3, 2, 1], rhs_shape=[])
bias_like_broadcast = BinaryShapeConfig(lhs_shape=[2, 7, 8, 8], rhs_shape=[7, 1, 1])

BinaryProfile = [
    ("Add_0D", tvm.relay.op.add, 'float32', base_0D),
    ("Add_4D", tvm.relay.op.add, 'float32', base_4D),
    ("Add_7D", tvm.relay.op.add, 'float32', base_6D),
    ("Add_Broadcast_scalar_4D", tvm.relay.op.add, 'float32', scalar_broadcast_6D),
    ("Add_BiasLike", tvm.relay.op.add, 'float32', bias_like_broadcast),
    ("Mul_BiasLike", tvm.relay.op.multiply, 'float32', bias_like_broadcast),
]


@requires_dnnl
@parametrized("profile", BinaryProfile)
def test_binary(profile):
    def generate_model(p, b_op_type, dtype):
        np.random.seed(0)
        bld = Builder()

        lhs = bld.arg(shape=p.lhs_shape, dtype=dtype, is_const=False)
        rhs = bld.arg(shape=p.rhs_shape, dtype=dtype, is_const=False)
        op = b_op_type(lhs, rhs)
        return bld.finalize(op)

    op_type, dtype, shape_p = profile
    ref_mod, args = generate_model(shape_p, op_type, dtype)
    mod = partition_for_dnnl(ref_mod)
    check_result(mod, ref_mod, args, tol=1e-10, atol=1e-10)
