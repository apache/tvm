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
"""Tests for functions in tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot.

Contains a few unit tests, followed by integration tests for common use cases. Note that we do not
run the generated code - we just make sure the strings match exactly.

Note that a *lot* of instruction reordering happens during compilation from C to assembly (by GCC or
Clang). I've verified that this instruction reordering happens correctly for all the functions here.
For more details on why the generated code is the way it is, see `tensordot_int16_impl`."""

import textwrap

from tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot import (
    _get_tensor_halfwords,
    _get_kernel_halfwords,
    tensordot_int16_impl,
)


def test_get_tensor_halfwords():
    """Tests the _get_tensor_halfwords helper function in tensordot.py.

    This function loads the logical indices of the data that will be stored in memory at the tensor
    pointer. See the function docstring for more details.
    """

    # fmt: off
    # A simple 3x3 depthwise convolution computing one output and with in_stride = 1. Note that each
    # row is padded with None at the end to make the rows word-aligned.
    assert _get_tensor_halfwords((48, 3, 3), 0, 1, 1) == [
        (0, 0), (0, 1), (0, 2), None,
        (1, 0), (1, 1), (1, 2), None,
        (2, 0), (2, 1), (2, 2), None
    ]

    # If the tensor width is odd, padding alternates before/after every row.
    assert _get_tensor_halfwords((49, 3, 3), 0, 1, 1) == [
        (0, 0), (0, 1), (0, 2), None,
        None, (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2), None
    ]

    # If we are computing multiple outputs, more tensor data becomes relevant.
    assert _get_tensor_halfwords((48, 3, 3), 0, 2, 1) == [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3)
    ]

    # If offset=1, relevant data starts one halfword after the kernel pointer.
    assert _get_tensor_halfwords((48, 3, 3), 1, 1, 1) == [
        None, (0, 0), (0, 1), (0, 2),
        None, (1, 0), (1, 1), (1, 2),
        None, (2, 0), (2, 1), (2, 2)
    ]

    # These adjustments can be (and often are) used together.
    assert _get_tensor_halfwords((49, 3, 3), 1, 2, 2) == [
        None, (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), None,
        None, (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)
    ]
    # fmt: on


def test_get_kernel_halfwords():
    """Tests the _get_kernel_halfwords helper function in tensordot.py.

    This function loads the logical indices of the data that will be stored in memory at the kernel
    pointer. See the function docstring for more details.
    """

    # fmt: off
    # Example of a kernel for a 3x3 depthwise convolution channel
    assert _get_kernel_halfwords((96, 3, 3), 0) == [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
        None,
    ]

    # Example of a kernel for a 1x1 regular convolution with 4 channels
    assert _get_kernel_halfwords((48, 1, 4), 1) == [
        None, (0, 0), (0, 1), (0, 2), (0, 3), None,
    ]
    # fmt: on


def test_write_3x3_depthwise_code():
    """This is the function that would be generated for a 1x4x48x48 NCHW input tensor with "SAME"
    padding. We are only computing one sum at once, so we don't need stride or output. Note that
    this is pretty inefficient - it would be much better to compute a few sums concurrently.

    When inlined, this code compiles (with armv7-a clang 11) into:

    tensordot_opt_x1_int16_w48_3x3_000(int*, int*, int*, int*, int*):
        ldr.w   lr, [r3]
        ldrd    r11, r4, [r1]
        ldrd    r5, r9, [r1, #96]
        ldrd    r10, r8, [r1, #192]
        ldm.w   r2, {r1, r6, r7}
        ldr.w   r12, [sp, #36]
        smlad   r1, r11, r1, lr
        smlabb  r1, r4, r6, r1
        smlatb  r1, r6, r5, r1
        ldrd    r3, r2, [r2, #12]
        smlatb  r1, r5, r7, r1
        smlatb  r1, r7, r9, r1
        smlad   r1, r10, r3, r1
        ldr.w   r3, [r12]
        smlabb  r1, r8, r2, r1
        smmul   r1, r3, r1
        ssat    r1, #8, r1, asr #8
        strh    r1, [r0]
    """
    _, code = tensordot_int16_impl(1, (48, 3, 3), (0, 0, 0), (1, 1))
    assert code == textwrap.dedent(
        """
    #ifndef TENSORDOT_OPT_X1_INT16_W48_3X3_000_EXISTS
    #define TENSORDOT_OPT_X1_INT16_W48_3X3_000_EXISTS
    #include <arm_acle.h>
    __attribute__((always_inline)) static inline int32_t tensordot_opt_x1_int16_w48_3x3_000(
        int16_t *output_arg, int16_t *tensor_arg, int16_t *kernel_arg,
        int32_t *bias, int32_t *scale
    ) {
      int32_t *output = output_arg;
      int32_t *tensor = tensor_arg;
      int32_t *kernel = kernel_arg;

      int32_t sum_0 = *bias;

      int32_t tensor__y00_x00__y00_x01 = tensor[0];
      int32_t tensor__y00_x02__unknown = tensor[1];
      int32_t tensor__y01_x00__y01_x01 = tensor[24];
      int32_t tensor__y01_x02__unknown = tensor[25];
      int32_t tensor__y02_x00__y02_x01 = tensor[48];
      int32_t tensor__y02_x02__unknown = tensor[49];

      int32_t kernel__y00_x00__y00_x01 = kernel[0];
      int32_t kernel__y00_x02__y01_x00 = kernel[1];
      int32_t kernel__y01_x01__y01_x02 = kernel[2];
      int32_t kernel__y02_x00__y02_x01 = kernel[3];
      int32_t kernel__y02_x02__unknown = kernel[4];

      sum_0 = __smlad(tensor__y00_x00__y00_x01, kernel__y00_x00__y00_x01, sum_0);
      sum_0 = __smlabb(tensor__y00_x02__unknown, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __smlabt(tensor__y01_x00__y01_x01, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __smlatb(tensor__y01_x00__y01_x01, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __smlabt(tensor__y01_x02__unknown, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __smlad(tensor__y02_x00__y02_x01, kernel__y02_x00__y02_x01, sum_0);
      sum_0 = __smlabb(tensor__y02_x02__unknown, kernel__y02_x02__unknown, sum_0);

      int32_t scale_val = *scale;
      int32_t requant_0 = (sum_0 * (int64_t) scale_val) >> 32;
      requant_0 = (requant_0 + 1) >> 1;
      requant_0 = __ssat(requant_0 + -128, 8);

      ((int16_t*) output)[0] = (int16_t) requant_0;
      return 0;
    }
    #endif
    """
    )


def test_odd_width_3x3_depthwise_strides_code():
    """This is the function that would be generated for a 1x4x48x48 NCHW input tensor with "SAME"
    padding and (2, 2) strides, being written into NHWC layout. The layout change is encoded by
    out_stride = 4. This is a common use case seen in MobileNetV1, among others.

    Note that despite the rows not being word-aligned, the *tensor pointer will always be word
    aligned (satisfying this requirement) since y_stride = 2."""

    _, code = tensordot_int16_impl(2, (49, 3, 3), (0, 0, 0), (2, 4))
    assert code == textwrap.dedent(
        """
    #ifndef TENSORDOT_OPT_X2_INT16_W49_3X3_000_2_4_EXISTS
    #define TENSORDOT_OPT_X2_INT16_W49_3X3_000_2_4_EXISTS
    #include <arm_acle.h>
    __attribute__((always_inline)) static inline int32_t tensordot_opt_x2_int16_w49_3x3_000_2_4(
        int16_t *output_arg, int16_t *tensor_arg, int16_t *kernel_arg,
        int32_t *bias, int32_t *scale
    ) {
      int32_t *output = output_arg;
      int32_t *tensor = tensor_arg;
      int32_t *kernel = kernel_arg;

      int32_t sum_0 = *bias, sum_1 = *bias;

      int32_t tensor__y00_x00__y00_x01 = tensor[0];
      int32_t tensor__y00_x02__y00_x03 = tensor[1];
      int32_t tensor__y00_x04__unknown = tensor[2];
      int32_t tensor__unknown__y01_x00 = tensor[24];
      int32_t tensor__y01_x01__y01_x02 = tensor[25];
      int32_t tensor__y01_x03__y01_x04 = tensor[26];
      int32_t tensor__y02_x00__y02_x01 = tensor[49];
      int32_t tensor__y02_x02__y02_x03 = tensor[50];
      int32_t tensor__y02_x04__unknown = tensor[51];

      int32_t kernel__y00_x00__y00_x01 = kernel[0];
      int32_t kernel__y00_x02__y01_x00 = kernel[1];
      int32_t kernel__y01_x01__y01_x02 = kernel[2];
      int32_t kernel__y02_x00__y02_x01 = kernel[3];
      int32_t kernel__y02_x02__unknown = kernel[4];

      sum_0 = __smlad(tensor__y00_x00__y00_x01, kernel__y00_x00__y00_x01, sum_0);
      sum_0 = __smlabb(tensor__y00_x02__y00_x03, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __smlatt(tensor__unknown__y01_x00, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __smlad(tensor__y01_x01__y01_x02, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __smlad(tensor__y02_x00__y02_x01, kernel__y02_x00__y02_x01, sum_0);
      sum_0 = __smlabb(tensor__y02_x02__y02_x03, kernel__y02_x02__unknown, sum_0);
      sum_1 = __smlad(tensor__y00_x02__y00_x03, kernel__y00_x00__y00_x01, sum_1);
      sum_1 = __smlabb(tensor__y00_x04__unknown, kernel__y00_x02__y01_x00, sum_1);
      sum_1 = __smlatt(tensor__y01_x01__y01_x02, kernel__y00_x02__y01_x00, sum_1);
      sum_1 = __smlad(tensor__y01_x03__y01_x04, kernel__y01_x01__y01_x02, sum_1);
      sum_1 = __smlad(tensor__y02_x02__y02_x03, kernel__y02_x00__y02_x01, sum_1);
      sum_1 = __smlabb(tensor__y02_x04__unknown, kernel__y02_x02__unknown, sum_1);

      int32_t scale_val = *scale;
      int32_t requant_0 = (sum_0 * (int64_t) scale_val) >> 32;
      requant_0 = (requant_0 + 1) >> 1;
      requant_0 = __ssat(requant_0 + -128, 8);
      int32_t requant_1 = (sum_1 * (int64_t) scale_val) >> 32;
      requant_1 = (requant_1 + 1) >> 1;
      requant_1 = __ssat(requant_1 + -128, 8);

      ((int16_t*) output)[0] = (int16_t) requant_0;
      ((int16_t*) output)[4] = (int16_t) requant_1;
      return 0;
    }
    #endif
    """
    )


def test_1x1x8_convolution_code():
    """This is the function that would be generated for a 1x48x48x8 NHWC input tensor under
    standard convolution with a 1x1 kernel. This is a common use case seen in MobileNetV1,
    among others. In this scenario, a very high amount of memory re-use means that summing
    four channels at once makes us faster."""

    _, code = tensordot_int16_impl(4, (48 * 8, 1, 8), (0, 0, 0), (8, 1))
    assert code == textwrap.dedent(
        """
    #ifndef TENSORDOT_OPT_X4_INT16_W384_1X8_000_8_1_EXISTS
    #define TENSORDOT_OPT_X4_INT16_W384_1X8_000_8_1_EXISTS
    #include <arm_acle.h>
    __attribute__((always_inline)) static inline int32_t tensordot_opt_x4_int16_w384_1x8_000_8_1(
        int16_t *output_arg, int16_t *tensor_arg, int16_t *kernel_arg,
        int32_t *bias, int32_t *scale
    ) {
      int32_t *output = output_arg;
      int32_t *tensor = tensor_arg;
      int32_t *kernel = kernel_arg;

      int32_t sum_0 = *bias, sum_1 = *bias, sum_2 = *bias, sum_3 = *bias;

      int32_t tensor__y00_x00__y00_x01 = tensor[0];
      int32_t tensor__y00_x02__y00_x03 = tensor[1];
      int32_t tensor__y00_x04__y00_x05 = tensor[2];
      int32_t tensor__y00_x06__y00_x07 = tensor[3];
      int32_t tensor__y00_x08__y00_x09 = tensor[4];
      int32_t tensor__y00_x0a__y00_x0b = tensor[5];
      int32_t tensor__y00_x0c__y00_x0d = tensor[6];
      int32_t tensor__y00_x0e__y00_x0f = tensor[7];
      int32_t tensor__y00_x10__y00_x11 = tensor[8];
      int32_t tensor__y00_x12__y00_x13 = tensor[9];
      int32_t tensor__y00_x14__y00_x15 = tensor[10];
      int32_t tensor__y00_x16__y00_x17 = tensor[11];
      int32_t tensor__y00_x18__y00_x19 = tensor[12];
      int32_t tensor__y00_x1a__y00_x1b = tensor[13];
      int32_t tensor__y00_x1c__y00_x1d = tensor[14];
      int32_t tensor__y00_x1e__y00_x1f = tensor[15];

      int32_t kernel__y00_x00__y00_x01 = kernel[0];
      int32_t kernel__y00_x02__y00_x03 = kernel[1];
      int32_t kernel__y00_x04__y00_x05 = kernel[2];
      int32_t kernel__y00_x06__y00_x07 = kernel[3];

      sum_0 = __smlad(tensor__y00_x00__y00_x01, kernel__y00_x00__y00_x01, sum_0);
      sum_0 = __smlad(tensor__y00_x02__y00_x03, kernel__y00_x02__y00_x03, sum_0);
      sum_0 = __smlad(tensor__y00_x04__y00_x05, kernel__y00_x04__y00_x05, sum_0);
      sum_0 = __smlad(tensor__y00_x06__y00_x07, kernel__y00_x06__y00_x07, sum_0);
      sum_1 = __smlad(tensor__y00_x08__y00_x09, kernel__y00_x00__y00_x01, sum_1);
      sum_1 = __smlad(tensor__y00_x0a__y00_x0b, kernel__y00_x02__y00_x03, sum_1);
      sum_1 = __smlad(tensor__y00_x0c__y00_x0d, kernel__y00_x04__y00_x05, sum_1);
      sum_1 = __smlad(tensor__y00_x0e__y00_x0f, kernel__y00_x06__y00_x07, sum_1);
      sum_2 = __smlad(tensor__y00_x10__y00_x11, kernel__y00_x00__y00_x01, sum_2);
      sum_2 = __smlad(tensor__y00_x12__y00_x13, kernel__y00_x02__y00_x03, sum_2);
      sum_2 = __smlad(tensor__y00_x14__y00_x15, kernel__y00_x04__y00_x05, sum_2);
      sum_2 = __smlad(tensor__y00_x16__y00_x17, kernel__y00_x06__y00_x07, sum_2);
      sum_3 = __smlad(tensor__y00_x18__y00_x19, kernel__y00_x00__y00_x01, sum_3);
      sum_3 = __smlad(tensor__y00_x1a__y00_x1b, kernel__y00_x02__y00_x03, sum_3);
      sum_3 = __smlad(tensor__y00_x1c__y00_x1d, kernel__y00_x04__y00_x05, sum_3);
      sum_3 = __smlad(tensor__y00_x1e__y00_x1f, kernel__y00_x06__y00_x07, sum_3);

      int32_t scale_val = *scale;
      int32_t requant_0 = (sum_0 * (int64_t) scale_val) >> 32;
      requant_0 = (requant_0 + 1) >> 1;
      requant_0 = __ssat(requant_0 + -128, 8);
      int32_t requant_1 = (sum_1 * (int64_t) scale_val) >> 32;
      requant_1 = (requant_1 + 1) >> 1;
      requant_1 = __ssat(requant_1 + -128, 8);
      int32_t requant_2 = (sum_2 * (int64_t) scale_val) >> 32;
      requant_2 = (requant_2 + 1) >> 1;
      requant_2 = __ssat(requant_2 + -128, 8);
      int32_t requant_3 = (sum_3 * (int64_t) scale_val) >> 32;
      requant_3 = (requant_3 + 1) >> 1;
      requant_3 = __ssat(requant_3 + -128, 8);

      int packed_res_0;
      __asm__ ("pkhbt %0, %1, %2, lsl #16" : "=r" (packed_res_0) : "r" (requant_0), "r" (requant_1));
      int packed_res_1;
      __asm__ ("pkhbt %0, %1, %2, lsl #16" : "=r" (packed_res_1) : "r" (requant_2), "r" (requant_3));
      output[0] = packed_res_0;
      output[1] = packed_res_1;
      return 0;
    }
    #endif
    """
    )


def test_3x3x3_offset_convolution_code():
    """This is the function that would be generated for a 1x96x96x3 NHWC input tensor under
    standard convolution with a 3x3x3 kernel - the first layer of MobileNetV1. This is special, as
    it means that every other kernel channel will not start on an even numbered halfword. We won't
    have this issue for the input tensor, as we will always compute two positions at a time.

    To solve this 'every other' issue, we will need two different version of this function to
    alternate between. This alternation will be handled in TIR scheduling. Here, we just test the
    version where the kernel is not word aligned.

    Also tests the requantize_shift and output_zero_point keyword args. These might be needed for
    some ResNet models (like image classification from MLPerf Tiny).
    """

    _, code = tensordot_int16_impl(
        1,
        (96 * 3, 3, 9),
        (1, 1, 1),
        (3, 1),
        requantize_shift=40,
        output_zero_point=4,
    )
    assert code == textwrap.dedent(
        """
    #ifndef TENSORDOT_OPT_X1_INT16_W288_3X9_111_EXISTS
    #define TENSORDOT_OPT_X1_INT16_W288_3X9_111_EXISTS
    #include <arm_acle.h>
    __attribute__((always_inline)) static inline int32_t tensordot_opt_x1_int16_w288_3x9_111(
        int16_t *output_arg, int16_t *tensor_arg, int16_t *kernel_arg,
        int32_t *bias, int32_t *scale
    ) {
      int32_t *output = output_arg;
      int32_t *tensor = tensor_arg;
      int32_t *kernel = kernel_arg;

      int32_t sum_0 = *bias;

      int32_t tensor__unknown__y00_x00 = tensor[0];
      int32_t tensor__y00_x01__y00_x02 = tensor[1];
      int32_t tensor__y00_x03__y00_x04 = tensor[2];
      int32_t tensor__y00_x05__y00_x06 = tensor[3];
      int32_t tensor__y00_x07__y00_x08 = tensor[4];
      int32_t tensor__unknown__y01_x00 = tensor[144];
      int32_t tensor__y01_x01__y01_x02 = tensor[145];
      int32_t tensor__y01_x03__y01_x04 = tensor[146];
      int32_t tensor__y01_x05__y01_x06 = tensor[147];
      int32_t tensor__y01_x07__y01_x08 = tensor[148];
      int32_t tensor__unknown__y02_x00 = tensor[288];
      int32_t tensor__y02_x01__y02_x02 = tensor[289];
      int32_t tensor__y02_x03__y02_x04 = tensor[290];
      int32_t tensor__y02_x05__y02_x06 = tensor[291];
      int32_t tensor__y02_x07__y02_x08 = tensor[292];

      int32_t kernel__unknown__y00_x00 = kernel[0];
      int32_t kernel__y00_x01__y00_x02 = kernel[1];
      int32_t kernel__y00_x03__y00_x04 = kernel[2];
      int32_t kernel__y00_x05__y00_x06 = kernel[3];
      int32_t kernel__y00_x07__y00_x08 = kernel[4];
      int32_t kernel__y01_x00__y01_x01 = kernel[5];
      int32_t kernel__y01_x02__y01_x03 = kernel[6];
      int32_t kernel__y01_x04__y01_x05 = kernel[7];
      int32_t kernel__y01_x06__y01_x07 = kernel[8];
      int32_t kernel__y01_x08__y02_x00 = kernel[9];
      int32_t kernel__y02_x01__y02_x02 = kernel[10];
      int32_t kernel__y02_x03__y02_x04 = kernel[11];
      int32_t kernel__y02_x05__y02_x06 = kernel[12];
      int32_t kernel__y02_x07__y02_x08 = kernel[13];

      sum_0 = __smlatt(tensor__unknown__y00_x00, kernel__unknown__y00_x00, sum_0);
      sum_0 = __smlad(tensor__y00_x01__y00_x02, kernel__y00_x01__y00_x02, sum_0);
      sum_0 = __smlad(tensor__y00_x03__y00_x04, kernel__y00_x03__y00_x04, sum_0);
      sum_0 = __smlad(tensor__y00_x05__y00_x06, kernel__y00_x05__y00_x06, sum_0);
      sum_0 = __smlad(tensor__y00_x07__y00_x08, kernel__y00_x07__y00_x08, sum_0);
      sum_0 = __smlatb(tensor__unknown__y01_x00, kernel__y01_x00__y01_x01, sum_0);
      sum_0 = __smlabt(tensor__y01_x01__y01_x02, kernel__y01_x00__y01_x01, sum_0);
      sum_0 = __smlatb(tensor__y01_x01__y01_x02, kernel__y01_x02__y01_x03, sum_0);
      sum_0 = __smlabt(tensor__y01_x03__y01_x04, kernel__y01_x02__y01_x03, sum_0);
      sum_0 = __smlatb(tensor__y01_x03__y01_x04, kernel__y01_x04__y01_x05, sum_0);
      sum_0 = __smlabt(tensor__y01_x05__y01_x06, kernel__y01_x04__y01_x05, sum_0);
      sum_0 = __smlatb(tensor__y01_x05__y01_x06, kernel__y01_x06__y01_x07, sum_0);
      sum_0 = __smlabt(tensor__y01_x07__y01_x08, kernel__y01_x06__y01_x07, sum_0);
      sum_0 = __smlatb(tensor__y01_x07__y01_x08, kernel__y01_x08__y02_x00, sum_0);
      sum_0 = __smlatt(tensor__unknown__y02_x00, kernel__y01_x08__y02_x00, sum_0);
      sum_0 = __smlad(tensor__y02_x01__y02_x02, kernel__y02_x01__y02_x02, sum_0);
      sum_0 = __smlad(tensor__y02_x03__y02_x04, kernel__y02_x03__y02_x04, sum_0);
      sum_0 = __smlad(tensor__y02_x05__y02_x06, kernel__y02_x05__y02_x06, sum_0);
      sum_0 = __smlad(tensor__y02_x07__y02_x08, kernel__y02_x07__y02_x08, sum_0);

      int32_t scale_val = *scale;
      int32_t requant_0 = (sum_0 * (int64_t) scale_val) >> 39;
      requant_0 = (requant_0 + 1) >> 1;
      requant_0 = __ssat(requant_0 + 4, 8);

      ((int16_t*) output)[1] = (int16_t) requant_0;
      return 0;
    }
    #endif
    """
    )
