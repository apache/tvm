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
"""Generates optimized code to compute a tensor dot product on ARMv7E-M.

This function can be used to tensorize many common operators including regular conv2d, depthwise
conv2d, and grouped conv2d for some data and kernel layouts. When for regular convolution, use data
layout HHWC and kernel layout OHWI. For depthwise convolution, use data layout data layout is NCHW
and kernel layout OIHW.
"""

from collections import namedtuple
from itertools import chain
import textwrap
from typing import Iterator, Tuple

SMLAInstruction = namedtuple("SMLAInstruction", ["instruction", "tensor_var", "kernel_var"])


def _get_c_function_name(split_size, dimensions, offsets, x_strides):
    """Generates a C function name for tensordot.

    We do not need a suffix, as the generated function will have an #include guard. Unlike other
    microTVM operators, _get_c_function_name is never called externally.
    """
    tensor_w, kernel_h, kernel_w = dimensions
    return (
        f"tensordot_opt_x{split_size}_int16_w{tensor_w}_"
        + f"{kernel_h}x{kernel_w}_"
        + "".join(map(str, offsets))
        + (f"_{x_strides[0]}_{x_strides[1]}" if split_size > 1 else "")
    )


def _init_biased_accumulators(split_size):
    """Generates code to load the bias into the accumulators.

    Addition is commutative, so we could add the bias before, during, or after performing our
    multiply-accumulate operations. Where we add the bias does not change the overflow behavior.

    Doing the bias add takes one cycle either way (if done at the beginning we can't use a SMULXY
    trick to set sum_i to zero for "free"). However, doing it at the beginning frees up a register,
    so we'll do it first.
    """
    assignments = map(lambda x: f"sum_{x:x} = *bias", range(split_size))
    joined_assignments = ", ".join(assignments)
    return f"int {joined_assignments};"


def _get_tensor_halfwords(dimensions, offset, split_size, in_stride) -> Iterator:
    tensor_w, kernel_h, kernel_w = dimensions

    split_max = (split_size - 1) * in_stride
    for y in range(kernel_h):
        if y * tensor_w % 2 + offset == 1:
            yield None
        for x in range(kernel_w + split_max):
            yield (y, x)
        if (y * tensor_w + kernel_w + split_max + offset) % 2 == 1:
            yield None


def _get_kernel_halfwords(dimensions, offset) -> Iterator:
    _, kernel_h, kernel_w = dimensions
    if offset == 1:
        yield None
    for y in range(kernel_h):
        for x in range(kernel_w):
            yield (y, x)
    if (kernel_h * kernel_w + offset) % 2 == 1:
        yield None


def _get_int16_alias(position) -> str:
    if not position:
        return "unknown"
    y, x = position
    return f"y{y:0>2x}_x{x:0>2x}"


def _load_tensor_vars(halfwords, tensor_w) -> Iterator[str]:
    assert len(halfwords) % 2 == 0
    offset = int(not bool(halfwords[0]))

    for i in range(0, len(halfwords), 2):
        var_name = "__".join(map(_get_int16_alias, halfwords[i : i + 2]))
        y, x = halfwords[i + 1] or halfwords[i]
        tensor_index = (y * tensor_w + x + offset) // 2
        yield f"int tensor__{var_name} = tensor[{tensor_index}];"


def _load_kernel_vars(halfwords) -> Iterator[str]:
    assert len(halfwords) % 2 == 0
    for i in range(0, len(halfwords), 2):
        var_name = "__".join(map(_get_int16_alias, halfwords[i : i + 2]))
        yield f"int kernel__{var_name} = kernel[{i // 2}];"


def _get_draft_macs(
    kernel_dims, tensor_halfwords, kernel_halfwords, offset
) -> Iterator[SMLAInstruction]:
    """Generates unrolled MAC instructions to compute one tensordot sum.

    Unrolling these loops increases code size a tiny bit (< 0.02 KB), but makes the generated code
    much faster. The generated code does not use SIMD instructions - they are added later by
    _apply_simd_optimizations.

    We return an iterator of SMLAInstruction named tuples. Returning an iterator lets us do
    optimizations by iterator chaining.
    """

    def get_var(y, x, halfwords) -> Tuple[str, str]:
        i = halfwords.index((y, x))
        if i % 2 == 0:
            return f"{_get_int16_alias((y, x))}__{_get_int16_alias(halfwords[i + 1])}", "b"
        return f"{_get_int16_alias(halfwords[i - 1])}__{_get_int16_alias((y, x))}", "t"

    kernel_h, kernel_w = kernel_dims
    for y in range(kernel_h):
        for x in range(kernel_w):
            tensor_var, tensor_half = get_var(y, x + offset, tensor_halfwords)
            kernel_var, kernel_half = get_var(y, x, kernel_halfwords)
            instruction = f"smla{tensor_half}{kernel_half}"
            yield SMLAInstruction(instruction, f"tensor__{tensor_var}", f"kernel__{kernel_var}")


def _apply_simd_optimizations(instruction_tuples) -> Iterator[SMLAInstruction]:
    """When possible, fuses single MACs into SIMD MAC instructions.

    The compiler cannot do this automatically, as calling __smlaxy forces the SMLAxy instruction to
    be used. This function takes as input an iterator of SMLAInstructions and returns an iterator of
    SMLAInstructions (possibly of different length).
    """
    curr_tuple = next(instruction_tuples, None)
    while curr_tuple:
        next_tuple = next(instruction_tuples, None)
        if not next_tuple:
            yield curr_tuple
            break

        if curr_tuple[1:] == next_tuple[1:]:
            if set([curr_tuple[0], next_tuple[0]]) == set(["smlatt", "smlabb"]):
                yield SMLAInstruction("smlad", *curr_tuple[1:])
                next_tuple = next(instruction_tuples, None)
            elif set([curr_tuple[0], next_tuple[0]]) == set(["smlatb", "smlabt"]):
                yield SMLAInstruction("smladx", *curr_tuple[1:])
                next_tuple = next(instruction_tuples, None)
            else:
                yield curr_tuple

        else:
            yield curr_tuple
        curr_tuple = next_tuple


def _expand_instruction_tuples(instruction_tuples, index) -> Iterator[str]:
    """Converts an iterator of SMLAInstructions into lines of C code.

    We want the compiler to re-order these with the memory loads, so we generate them as a series of
    calls to instruction aliases instead of as a single `asm` block.
    """

    for instruction, op1, op2 in instruction_tuples:
        assert "smla" in instruction

        # We call the instruction using the Arm C Language Extensions. Using ACLE gives better
        # cross-compiler compatibility than using __builtin functions.
        yield f"sum_{index} = __{instruction}({op1}, {op2}, sum_{index});"


def _requantize_sums(num_sums, requantize_shift, output_zero_point) -> Iterator[str]:
    """Generates code to requantize the accumulator values.

    The generated code does not use floating point instructions, as it simulates floating point
    multiplication with an a int64 multiply + shift. The bias is added at the beginning, so we can
    skip doing it now. The shift is hard-coded, as this saves a few cycles without hurting accuracy
    in "most" cases.

    It's *possible* we could save one more cycle here by pre-multiplying the bias with the
    requantize multiplier, and then doing the bias addition and shift in the same cycle (via <op2>).
    However, it's complicated and only saves one cycle.

    It's also worth noting the SSAT16 operation doesn't help us here. The data isn't stored as two
    halfwords in a word, and rearrainging it would take at least one cycle. Two SSAT operations is
    just as good.

    Calling __ssat directly is a little bit gross, but GCC and Clang are unreliable about compiling
    other ways of writing this. Both the multiply + shift and shift + saturation combine to one
    instruction each.
    """

    yield "int scale_val = *scale;"
    for i in range(num_sums):
        yield f"int requant_{i} = (sum_{i} * (long long) scale_val) >> {requantize_shift - 1};"
        yield f"requant_{i} = (requant_{i} + 1) >> 1;"
        yield f"requant_{i} = __ssat(requant_{i} + {output_zero_point}, 8);"


def _write_sums_to_memory(num_sums, offset, stride) -> Iterator[str]:
    """Generates code to write the requantized sums to memory.

    Note - halfword packing here *does* help. It seems
    like it wouldn't, as doing two pipelined int16 stores takes two cycles - the same as halfword
    packing plus a pipelined int32 store. We still do the int16 stores when there is an output
    stride, though.

    However, this lets the compiler re-order instructions to better preserve memory, as it doesn't
    like breaking apart the store instructions (as this messes up pipelining).
    """

    if stride > 1:
        for i in range(num_sums):
            yield f"((short*) output)[{i * stride + offset}] = (short) requant_{i};"

    else:
        num_halfwords = (num_sums - offset) // 2
        for i in range(num_halfwords):
            index = 2 * i + offset
            yield f"int packed_res_{i} = requant_{index} + (requant_{index + 1} << 16);"

        if offset == 1:
            yield "((short*) output)[1] = (short) requant_0;"

        for i in range(num_halfwords):
            yield f"output[{offset + i}] = packed_res_{i};"

        if (offset + num_sums) % 2 == 1:
            yield f"((short*) output)[{num_halfwords * 2}] = (short) requant_{num_halfwords * 2};"


def tensordot_int16_impl(
    split_size: int,
    dimensions: Tuple[int, int, int],
    offsets: Tuple[int, int, int],
    x_strides: Tuple[int, int],
    requantize_shift: int = 33,
    output_zero_point: int = -128,
) -> Tuple[str, str]:
    """Generates code to compute a tensor dot product with requantization.

    The generated function takes pointers to the output, tensor, and kernel as input. All pointers
    must be word aligned. Only works with `int16` data type. The generated code is optimized for the
    ARMv7E-M architecture.

    Parameters
    ----------
    split_size: int
        The number of tensordot values to compute in this function. Computing more than one at once
        makes us much faster by reducing how often overlapping data is loaded. However, setting this
        too high causes us to run out of registers and need to store data on the stack. We should
        autotune this, but split_size=2 is usually OK.

    dimensions: Tuple[int, int, int]
        The dimensions of each tensordot operation. dimensions[1] and dimensions[2] are the height
        and width of the kernel, respectively. dimensions[0] is the width of the data tensor, which
        is usually larger than the kernel.

    offsets: Tuple[int, int, int]
        Each value is 0 or 1, and represents how far after the given data, kernel, and output
        pointers (respectively) we should start reading/writing. This prevents us from having to
        check if each pointer is aligned or unaligned at runtime, making us faster.

    x_strides: Tuple[int, int]
        The distance (in halfwords) between the start of each input tensor, and where to write each
        output result respectively. Only used when split_size > 1.

    requantize_shift: int
        The distance to right shift after multiplying by the requantization scale. Defaults to 33,
        as this lets us skip a shift operation.

    outout_zero_point: int
        The output zero point, which will be subtracted after scale multiplication but before
        clipping. Defaults to -128, as most models always use this.

    Returns
    -------
    func_name, func_code: Tuple[str, str]
        The name and source code of the generated function.
    """
    function_name = _get_c_function_name(split_size, dimensions, offsets, x_strides)
    tensor_w, kernel_h, kernel_w = dimensions
    tensor_offset, kernel_offset, output_offset = offsets
    assert tensor_offset < 2 and kernel_offset < 2 and output_offset < 2
    in_stride, out_stride = x_strides

    tensor_halfwords = list(_get_tensor_halfwords(dimensions, tensor_offset, split_size, in_stride))
    kernel_halfwords = list(_get_kernel_halfwords(dimensions, kernel_offset))
    load_tensor_lines = _load_tensor_vars(tensor_halfwords, tensor_w)
    load_kernel_lines = _load_kernel_vars(kernel_halfwords)

    def gen_single_loop_macs(index):
        draft_macs_iter = _get_draft_macs(
            (kernel_h, kernel_w), tensor_halfwords, kernel_halfwords, index * in_stride
        )
        draft_macs_iter = _apply_simd_optimizations(draft_macs_iter)
        return _expand_instruction_tuples(draft_macs_iter, index)

    multiply_acc_lines = chain.from_iterable(gen_single_loop_macs(i) for i in range(split_size))
    requantize_lines = _requantize_sums(
        split_size, requantize_shift=requantize_shift, output_zero_point=output_zero_point
    )
    write_out_lines = _write_sums_to_memory(split_size, output_offset, out_stride)

    def insert_lines(lines):
        return ("\n" + " " * 10).join(lines)

    # It's very common for one model to have different layers that use identical tensordot
    # functions. To prevent function re-definition errors, we need an #include guard. This is better
    # than adding a random suffix, as it saves flash memory.
    code = textwrap.dedent(
        f"""
        #ifndef {function_name.upper()}_EXISTS
        #define {function_name.upper()}_EXISTS
        #include <arm_acle.h>
        __attribute__((always_inline)) static inline int {function_name}(
            int *output, int *tensor, int *kernel, int *bias, int *scale
        ) {{
          {_init_biased_accumulators(split_size)}

          {insert_lines(load_tensor_lines)}

          {insert_lines(load_kernel_lines)}

          {insert_lines(multiply_acc_lines)}

          {insert_lines(requantize_lines)}

          {insert_lines(write_out_lines)}
          return 0;
        }}
        #endif
        """
    )
    return (function_name, code)
