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
# pylint: disable=invalid-name,unused-argument
"""Strategy and AlterOpLayout functions of MLAS operators"""
import tvm
from tvm import relay, topi
from tvm.te.hybrid import script
from .strategy import wrap_topi_schedule
from . import op as reg


# Mlas_matmul
# Mlas_matmul strategy
@tvm.target.override_native_generic_func("mlas_matmul_strategy")
def mlas_matmul_strategy(attrs, inputs, out_type, target):
    """mlas_matmul generic strategy"""
    return None


@mlas_matmul_strategy.register(["cpu", "arm_cpu"])
def mlas_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """mlas_matmul strategy"""
    strategy = reg.OpStrategy()

    def wrap_compute_mlas_matmul(topi_compute):
        """wrap mlas_matmul topi compute"""

        def _compute_mlas_matmul(attrs, inputs, out_type):
            args = [inputs[0], inputs[1], attrs.packb, attrs.K, attrs.N]
            return [topi_compute(*args)]

        return _compute_mlas_matmul

    strategy.add_implementation(
        wrap_compute_mlas_matmul(topi.mlas_matmul),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="mlas_matmul",
        plevel=1,
    )
    return strategy


reg.register_strategy("mlas_matmul", mlas_matmul_strategy)
reg.register_pattern("mlas_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# Mlas_matmul AlterOpLayout
@tvm.target.generic_func
def batch_matmul_alter_layout(attrs, inputs, tinfos, out_type):
    """Change batch_matmul layout."""
    # not to change by default
    return None


@batch_matmul_alter_layout.register(["cpu", "arm_cpu"])
def _alter_batch_matmul_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    if (
        "mlas" in target.libs
        and tinfos[0].dtype == "float32"
        and tinfos[1].dtype == "float32"
        and out_type.dtype == "float32"
    ):
        # mlas is only used for static tensors
        if not (
            any([isinstance(dim, tvm.tir.Any) for dim in tinfos[0].shape])
            or any([isinstance(dim, tvm.tir.Any) for dim in tinfos[1].shape])
        ):
            # if matrix B is constant, use packed matmul
            if isinstance(inputs[1], relay.expr.Constant):
                b_shape = inputs[1].data.shape
                assert len(b_shape) == 3
                batch, N, K = b_shape[0], b_shape[1], b_shape[2]
                # batch_B must be 1
                if batch == 1:
                    packed_b = relay.op.mlas_packb(inputs[1], K, N)
                    output = relay.op.mlas_matmul(inputs[0], packed_b, True, K, N)
                    return output
            # if matrix A, B are not constant and no other libs are enabled, use normal matmul
            if not any([item in target.libs for item in ["mkl", "clbas", "mkldnn"]]):
                return relay.op.mlas_matmul(inputs[0], inputs[1], False)
    return None


@reg.register_alter_op_layout("nn.batch_matmul")
def alter_op_layout_dense(attrs, inputs, tinfos, out_type):
    """Alternate the layout of batch_matmul"""
    return batch_matmul_alter_layout(attrs, inputs, tinfos, out_type)


# Dense
# Dense strategy
@tvm.target.override_native_generic_func("mlas_packb_strategy")
def mlas_packb_strategy(attrs, inputs, out_type, target):
    """mlas_packb generic strategy"""
    strategy = reg.OpStrategy()

    def wrap_mlas_packb(topi_compute):
        """Wrap mlas_packb topi compute"""

        def _compute_mlas_packb(attrs, inputs, _):
            return [topi_compute(inputs[0], attrs.K, attrs.N, attrs.size, attrs.transb)]

        return _compute_mlas_packb

    strategy.add_implementation(
        wrap_mlas_packb(topi.mlas_packb),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="mlas_packb",
    )
    return strategy


reg.register_strategy("mlas_packb", mlas_packb_strategy)

# Dense AlterOpLayout
# See tvm.topi.x86.dense_alter_op


@script
def _mlas_matmul_shape_func(tensor_a_shape, tensor_b_shape):
    out = output_tensor((tensor_a_shape.shape[0],), "int64")
    if tensor_a_shape.shape[0] == 3:
        out[0] = tensor_a_shape[0]
        out[1] = tensor_a_shape[1]
        out[2] = tensor_b_shape[1]
    else:
        out[0] = tensor_a_shape[0]
        out[1] = tensor_b_shape[0]
    return out


@script
def _mlas_matmul_packb_shape_func(tensor_a_shape, N):
    out = output_tensor((tensor_a_shape.shape[0],), "int64")
    if tensor_a_shape.shape[0] == 3:
        out[0] = tensor_a_shape[0]
        out[1] = tensor_a_shape[1]
        out[2] = N
    else:
        out[0] = tensor_a_shape[0]
        out[1] = N
    return out


@reg.register_shape_func("mlas_matmul", False)
def matmul_shape_func(attrs, inputs, _):
    """Shape function for matmul op."""
    if attrs.packb:
        return [_mlas_matmul_packb_shape_func(inputs[0], tvm.tir.expr.IntImm("int64", attrs.N))]
    return [_mlas_matmul_shape_func(inputs[0], inputs[1])]
