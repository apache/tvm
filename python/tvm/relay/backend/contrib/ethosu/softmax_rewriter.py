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
"""SoftmaxRewriter for legalization Softmax operation."""
import math

import numpy as np
from ethosu.vela import fp_math, scaling

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu import op as ethosu_ops
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard
from tvm.relay.op.contrib import ethosu as ethosu_patterns


class SoftmaxRewriter(DFPatternCallback):
    """This rewriting converts Softmax operation into a sequence of operations as in Vela."""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.params_class = ethosu_patterns.SoftMaxParams
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.SoftMaxParams.composite_name})
        )(None)

    def generate_exp_table(self, input_scale):
        """Generate a LUT table for exponential function.

        Parameters
        ----------
        input_scale : float
            The scale for input.

        Returns
        -------
        lut : tvm.relay.expr.Constant
            LUT table for exponential function.
        """
        beta = 1.0
        integer_bits = 5
        total_signed_bits = 31
        # Calculate scaling
        real_beta = min(
            np.double(beta) * np.double(input_scale) * (1 << (31 - integer_bits)),
            np.double((1 << 31) - 1.0),
        )
        scale, shift = scaling.quantise_scale(real_beta)
        shift = 31 - shift
        diff_min = -1.0 * math.floor(
            1.0
            * ((1 << integer_bits) - 1)
            * (1 << (total_signed_bits - integer_bits))
            / (1 << shift)
        )
        # Generate the exp LUT
        lut = []
        for x in range(256):
            input_diff = x - 255
            if input_diff >= diff_min:
                rescale = fp_math.saturating_rounding_mul32(input_diff * (1 << shift), scale)
                lut.append(fp_math.exp_on_negative_values(rescale))
            else:
                lut.append(0)
        res = np.array(lut, dtype="int32")
        return relay.const(res)

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = self.params_class(post.op.body)
        quant_min = -128
        quant_max = 127

        ifm = post.args[0]
        ifm_dtype = ifm.checked_type.dtype
        bhw = np.prod(params.ifm.shape[:-1])
        depth = params.ifm.shape[-1]

        # The calculation of Softmax is similar to that in Vela
        # https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/3.7.0/ethosu/vela/softmax.py#230
        # PASS 0 - Depthwise Maxpool
        # reshape for depthwise maxpool
        ifm = relay.reshape(ifm, (1, bhw, depth, 1))
        lut = relay.const([], dtype="int32")
        depthwise_maxpool = ethosu_ops.ethosu_pooling(
            ifm=ifm,
            lut=lut,
            pooling_type="MAX",
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            pool_shape=(1, depth),
            ofm_channels=1,
            ofm_dtype=ifm_dtype,
        )

        # PASS 1 - Sub+LUT(exp)
        # move all data along the height axis, except channels
        ifm = relay.reshape(ifm, (1, bhw, 1, depth))
        exp_lut = self.generate_exp_table(float(params.ifm.q_params.scale_f32))
        ifm_exp = ethosu_ops.ethosu_binary_elementwise(
            ifm=ifm,
            ifm2=depthwise_maxpool,
            lut=exp_lut,
            operator_type="SUB",
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ifm2_scale=0.0,
            ifm2_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=1.0,
            ofm_zero_point=quant_max,
            ifm_channels=depth,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="LUT",
            clip_min=-255,
            clip_max=0,
        )

        # PASS 2 - SHR
        shr_const = relay.const(np.full([1, 1, 1, 1], 12, dtype="int32"))
        shr = ethosu_ops.ethosu_binary_elementwise(
            ifm=ifm_exp,
            ifm2=shr_const,
            lut=lut,
            operator_type="SHR",
            ifm_scale=1.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ifm_channels=params.ifm.shape[-1],
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
            rounding_mode="NATURAL",
        )

        # PASS 3 - Reduce sum
        sum_of_exp = ethosu_ops.ethosu_pooling(
            ifm=shr,
            lut=lut,
            pooling_type="SUM",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            pool_shape=(1, 1),
            ofm_channels=1,
            upscale="NONE",
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 4 - CLZ
        headroom_plus_one = ethosu_ops.ethosu_unary_elementwise(
            ifm=sum_of_exp,
            lut=lut,
            operator_type="CLZ",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_channels=1,
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 5 - Sub
        headroom_offset_const = relay.const(np.full([1, bhw, 1, 1], 35, dtype="int32"))
        right_shift = ethosu_ops.ethosu_binary_elementwise(
            ifm=headroom_offset_const,
            ifm2=headroom_plus_one,
            lut=lut,
            operator_type="SUB",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 6 - Sub
        one_const = relay.const(np.full([1, 1, 1, 1], 1, dtype="int32"))
        headroom = ethosu_ops.ethosu_binary_elementwise(
            ifm=headroom_plus_one,
            ifm2=one_const,
            lut=lut,
            operator_type="SUB",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 7 - SHL
        shifted_sum = ethosu_ops.ethosu_binary_elementwise(
            ifm=sum_of_exp,
            ifm2=headroom,
            lut=lut,
            operator_type="SHL",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 8 - Sub
        shifted_one_const = relay.const(np.full([1, 1, 1, 1], 1 << 30, dtype="int32"))
        shifted_sum_minus_one = ethosu_ops.ethosu_binary_elementwise(
            ifm=shifted_sum,
            ifm2=shifted_one_const,
            lut=lut,
            operator_type="SUB",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 9 - SHL
        shifted_sum_minus_one = ethosu_ops.ethosu_binary_elementwise(
            ifm=shifted_sum_minus_one,
            ifm2=one_const,
            lut=lut,
            operator_type="SHL",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ifm.q_params.zero_point),
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 10 - Add
        f0_one_const = relay.const(np.full([1, 1, 1, 1], (1 << 31) - 1, dtype="int32"))
        half_denominator = ethosu_ops.ethosu_binary_elementwise(
            ifm=shifted_sum_minus_one,
            ifm2=f0_one_const,
            lut=lut,
            operator_type="ADD",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=1.0,
            ofm_zero_point=0,
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
            use_rescale=True,
            rescale_scale=1,
            rescale_shift=1,
        )

        # PASS 11 - Mul
        neg_32_over_17_const = relay.const(np.full([1, 1, 1, 1], -1010580540, dtype="int32"))
        rescaled = ethosu_ops.ethosu_binary_elementwise(
            ifm=half_denominator,
            ifm2=neg_32_over_17_const,
            lut=lut,
            operator_type="MUL",
            ifm_scale=1.0,
            ifm_zero_point=0,
            ifm2_scale=1.0,
            ifm2_zero_point=0,
            ofm_scale=2.0,
            ofm_zero_point=0,
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 12 - Add
        const_48_over_17_const = relay.const(np.full([1, 1, 1, 1], 1515870810, dtype="int32"))
        rescale_w_offset = ethosu_ops.ethosu_binary_elementwise(
            ifm=rescaled,
            ifm2=const_48_over_17_const,
            lut=lut,
            operator_type="ADD",
            ifm_scale=2.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=1.0,
            ofm_zero_point=0,
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        nr_x = rescale_w_offset
        f2_one_const = relay.const(np.full([1, bhw, 1, 1], 1 << 29, dtype="int32"))
        four_const = relay.const(np.full([1, 1, 1, 1], 4, dtype="int32"))
        for _ in range(3):
            # PASS 13, 18, 23 - Mul
            half_denominator_times_x = ethosu_ops.ethosu_binary_elementwise(
                ifm=nr_x,
                ifm2=half_denominator,
                lut=lut,
                operator_type="MUL",
                ifm_scale=1.0,
                ifm_zero_point=0,
                ifm2_scale=1.0,
                ifm2_zero_point=0,
                ofm_scale=2.0,
                ofm_zero_point=0,
                ifm_channels=1,
                ifm2_channels=1,
                reversed_operands=False,
                ofm_dtype="int32",
                activation="CLIP",
                clip_min=quant_min,
                clip_max=quant_max,
            )

            # PASS 14, 19, 24 - Sub
            one_minus_half_denomin_times_x = ethosu_ops.ethosu_binary_elementwise(
                ifm=f2_one_const,
                ifm2=half_denominator_times_x,
                lut=lut,
                operator_type="SUB",
                ifm_scale=2.0,
                ifm_zero_point=0,
                ifm2_scale=0.0,
                ifm2_zero_point=0,
                ofm_scale=1.0,
                ofm_zero_point=0,
                ifm_channels=1,
                ifm2_channels=1,
                reversed_operands=False,
                ofm_dtype="int32",
                activation="CLIP",
                clip_min=quant_min,
                clip_max=quant_max,
            )

            # PASS 15, 20, 25 - Mul
            to_rescale = ethosu_ops.ethosu_binary_elementwise(
                ifm=nr_x,
                ifm2=one_minus_half_denomin_times_x,
                lut=lut,
                operator_type="MUL",
                ifm_scale=1.0,
                ifm_zero_point=0,
                ifm2_scale=1.0,
                ifm2_zero_point=0,
                ofm_scale=2.0,
                ofm_zero_point=0,
                ifm_channels=1,
                ifm2_channels=1,
                reversed_operands=False,
                ofm_dtype="int32",
                activation="CLIP",
                clip_min=quant_min,
                clip_max=quant_max,
            )

            # PASS 16, 21, 26 - Mul
            to_add = ethosu_ops.ethosu_binary_elementwise(
                ifm=to_rescale,
                ifm2=four_const,
                lut=lut,
                operator_type="MUL",
                ifm_scale=2.0,
                ifm_zero_point=0,
                ifm2_scale=0.0,
                ifm2_zero_point=0,
                ofm_scale=0.0,
                ofm_zero_point=int(params.ifm.q_params.zero_point),
                ifm_channels=1,
                ifm2_channels=1,
                reversed_operands=False,
                ofm_dtype="int32",
                activation="CLIP",
                clip_min=quant_min,
                clip_max=quant_max,
            )

            # PASS 17, 22, 27 - Add
            nr_x = ethosu_ops.ethosu_binary_elementwise(
                ifm=nr_x,
                ifm2=to_add,
                lut=lut,
                operator_type="ADD",
                ifm_scale=1.0,
                ifm_zero_point=0,
                ifm2_scale=0.0,
                ifm2_zero_point=0,
                ofm_scale=1.0,
                ofm_zero_point=0,
                ifm_channels=1,
                ifm2_channels=1,
                reversed_operands=False,
                ofm_dtype="int32",
                activation="CLIP",
                clip_min=quant_min,
                clip_max=quant_max,
            )

        # PASS 28 - Mul
        two_const = relay.const(np.full([1, 1, 1, 1], 2, dtype="int32"))
        scale_factor = ethosu_ops.ethosu_binary_elementwise(
            ifm=nr_x,
            ifm2=two_const,
            lut=lut,
            operator_type="MUL",
            ifm_scale=1.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=1.0,
            ofm_zero_point=0,
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 29 - Mul
        scaled_exp = ethosu_ops.ethosu_binary_elementwise(
            ifm=ifm_exp,
            ifm2=scale_factor,
            lut=lut,
            operator_type="MUL",
            ifm_scale=1.0,
            ifm_zero_point=0,
            ifm2_scale=1.0,
            ifm2_zero_point=0,
            ofm_scale=2.0,
            ofm_zero_point=0,
            ifm_channels=depth,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int32",
            activation="CLIP",
            clip_min=quant_min,
            clip_max=quant_max,
        )

        # PASS 30 - SHR
        shr30_op = ethosu_ops.ethosu_binary_elementwise(
            ifm=scaled_exp,
            ifm2=right_shift,
            lut=lut,
            operator_type="SHR",
            ifm_scale=2.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            ifm_channels=depth,
            ifm2_channels=1,
            reversed_operands=False,
            rounding_mode="NATURAL",
            ofm_dtype=ifm_dtype,
        )

        reshape = relay.reshape(shr30_op, params.ofm.shape)
        return reshape
