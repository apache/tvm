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
"""Computes a "jumpy tensordot" operator, which can be used to tensorize many common operators
including regular conv2d, depthwise conv2d, and grouped conv2d provided the data and kernel layouts
are the optimal ones. When groups=1, the optimal data layout is NHWC and kernel layout is OHWI. When
this is a depthwise convolution, the optimal data layout is NCHW and kernel layout is OIHW."""

from itertools import chain
import textwrap
from typing import Iterator, Tuple

import numpy as np

from tvm import te, tir


def get_c_function_name(split_size, dimensions, offsets, x_strides):
    """Gets the C function name of the tensordot function."""
    tensor_w, kernel_h, kernel_w = dimensions
    return (
        f"tensordot_opt_x{split_size}_int16_w{tensor_w}_"
        + f"{kernel_h}x{kernel_w}_"
        + "".join(map(str, offsets))
        + (f"_{x_strides[0]}_{x_strides[1]}" if split_size > 1 else "")
    )

def _is_pow_2(number):
    """Checks if `number` is a power of `2`."""
    return number & (number - 1) == 0 and number > 0


def _count_factorization_2s(number):
    """Returns the number of times `2` appears in the factorization of `number`."""
    assert isinstance(number, int)
    count = 0
    while number % 2 == 0:
        number // 2
        count += 1
    return count


def _init_biased_accumulators(split_size):
    """Addition is commutative, so we could add the bias before, during, or after performing our
    multiply-accumulate operations. It "costs" one cycle either way - if done at the beginning we
    can't use our SMULXY trick to set sum_i to zero for "free", and if done at the end it doesn't
    combine with anything. However, doing it at the beginning frees up a register/prevents needing
    to do a stack push/pop, so we'll do it first."""
    var_names = map(lambda x: f"sum_{x:x}", range(split_size))
    joined_var_names = ", ".join(var_names)
    return f"int {joined_var_names} = *bias;"


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
    _tensor_w, kernel_h, kernel_w = dimensions
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
    for i in range(0, len(halfwords), 2):
        var_name = "__".join(map(_get_int16_alias, halfwords[i : i + 2]))
        y, x = halfwords[i + 1] or halfwords[i]
        tensor_index = (y * tensor_w + x) // 2
        yield f"int tensor__{var_name} = tensor[{tensor_index}];"


def _load_kernel_vars(halfwords) -> Iterator[str]:
    assert len(halfwords) % 2 == 0
    for i in range(0, len(halfwords), 2):
        var_name = "__".join(map(_get_int16_alias, halfwords[i : i + 2]))
        yield f"int kernel__{var_name} = kernel[{i // 2}];"


def _get_draft_macs(kernel_dims, tensor_halfwords, kernel_halfwords, offset) -> Iterator[Tuple]:
    def get_var(y, x, halfwords):
        i = halfwords.index((y, x))
        if i % 2 == 0:
            return f"{_get_int16_alias((y, x))}__{_get_int16_alias(halfwords[i + 1])}", "b"
        else:
            return f"{_get_int16_alias(halfwords[i - 1])}__{_get_int16_alias((y, x))}", "t"

    kernel_h, kernel_w = kernel_dims
    for y in range(kernel_h):
        for x in range(kernel_w):
            tensor_var, tensor_half = get_var(y, x + offset, tensor_halfwords)
            kernel_var, kernel_half = get_var(y, x, kernel_halfwords)
            yield f"smla{tensor_half}{kernel_half}", f"tensor__{tensor_var}", f"kernel__{kernel_var}"


def _apply_simd_optimizations(instruction_tuples) -> Iterator[Tuple]:
    curr_tuple = next(instruction_tuples, None)
    while curr_tuple:
        next_tuple = next(instruction_tuples, None)
        if not next_tuple:
            yield curr_tuple
            break

        if curr_tuple[1:] == next_tuple[1:]:
            if set([curr_tuple[0], next_tuple[0]]) == set(["smlatt", "smlabb"]):
                yield "smlad", *curr_tuple[1:]
                next_tuple = next(instruction_tuples, None)
            elif set([curr_tuple[0], next_tuple[0]]) == set(["smlatb", "smlabt"]):
                yield "smladx", *curr_tuple[1:]
                next_tuple = next(instruction_tuples, None)
            else:
                yield curr_tuple

        else:
            yield curr_tuple
        curr_tuple = next_tuple


NO_ACC_PREFIX_CONVERSIONS = {
    "smlad": "smuad",
    "smladx": "smuadx",
    "smlatt": "smultt",
    "smlatb": "smultb",
    "smlabt": "smulbt",
    "smlabb": "smulbb",
}


#def _no_first_accumulate(instruction_tuples) -> Iterator[Tuple]:
#    ins, op1, op2 = next(instruction_tuples)
#    yield NO_ACC_PREFIX_CONVERSIONS[ins], op1, op2
#    for instruction_tuple in instruction_tuples:
#        yield instruction_tuple


def _expand_instruction_tuples(instruction_tuples, index) -> Iterator[str]:
    """Converts a series of (instruction, var1, var2) tuples into lines of C code. Should be simple,
    but we need to work around a series of cryptic bugs while ensuring the compiler makes certain
    optimizations.

    1. Ideally, we would call __builtin_arm functions instead of including inline assembly, as this
       is easier to read and more future proof. However:
        a. Arm GCC apparently *forgot* to include `__builtin_arm_smlabt`, even though
           `__builtin_arm_smlatt`, `__builtin_arm_smlatb`, `__builtin_arm_smlad` and so on all
           exist. These work as expected on Clang - the issue is GCC only.

        b. Calling `__builtin_arm_smlatt` (and `smlatb` and `smlabb`) works fine on real devices.
           However, calling these builtins causes the Corstone300 simulator to freeze and stall. I
           have no clue on why this is - wouldn't these builtins be compiled to assembly? - yet it
           occurs consistently.


    2. Ideally, the compiler would know that the first multiply instruction should *not* accumulate,
       and would automatically replace it with an otherwise identical but non-accumulating
       instruction. Doing this saves us one cycle, as we don't need to load a zero into sum_i.
       However, the compiler (understandably) does not like overwriting instructions we explicitly
       as for, so we must do this ourselves.

    3. Ideally, since we're going to emit several lines of assembly code, we would do it in a single
       `asm` block. However, we *want* the compiler to reorder the instructions and interleave them
       with memory loads, and it can only do this if we specify the instructions as individual non-
       volatile memory loads.
    """
    for instruction, op1, op2 in instruction_tuples:
        if "smla" in instruction:
            if instruction == "smlabt":
                yield f"sum_{index} = __builtin_arm_smlatb({op2}, {op1}, sum_{index});"
            else:
                yield f"sum_{index} = __builtin_arm_{instruction}({op1}, {op2}, sum_{index});"

        else:
            yield f'asm ("{instruction} %0, %1, %2" : "=r" (sum_{index}) : "r" ({op1}), "r" ({op2}));'

def _requantize_sums(num_sums) -> Iterator[str]:
    """Simulates multiplying by the float32 requantization scale by doing a int64 multiply + shift,
    which is much faster. The bias is added at the beginning, so we can skip doing it now. The shift
    is hard-coded, as this saves a few cycles without hurting accuracy in "most" cases.

    It's *possible* we could save one more cycle here by pre-multiplying the bias with the
    requantize multiplier, and then doing the bias addition and shift in the same cycle (via <op2>).
    However, it's complicated and only saves one cycle.

    It's also worth noting the SSAT16 operation doesn't help us here. The data isn't stored as two
    halfwords in a word, and rearrainging it would take at least one cycle. Two SSAT operations is
    just as good.

    Lastly, calling __builtin_arm_ssat is a little bit gross, but GCC and Clang are unreliable about
    compiling other ways of writing this. Both the multiply + shift and shift + saturation combine
    to one instruction each."""

    yield "int requantize_multiplier = *requant_scale;"
    for i in range(num_sums):
        yield f"int requant_{i} = (sum_{i} * (long long) requantize_multiplier) >> 32;"
        yield f"requant_{i} = __builtin_arm_ssat(requant_{i} >> 8, 8);"


def _write_sums_to_memory(num_sums, offset, stride) -> Iterator[str]:
    """Writes the requantized sums to memory. Note - halfword packing here *does* help. It seems
    like it wouldn't, as doing two pipelined int16 stores takes two cycles - the same as halfword
    packing plus a pipelined int32 store. We still do the int16 stores when there is an output
    stride, though.

    However, this lets the compiler re-order instructions to better preserve memory, as it doesn't
    like breaking apart the store instructions (as this messes up pipelining)."""

    if stride > 1:
        for i in range(num_sums):
            yield f"((short*) output)[{i * stride + offset}] = (short) requant_{i};"

    else:
        num_halfwords = (num_sums - offset) // 2
        for i in range(num_halfwords):
            index = 2 * i + offset
            yield f"int packed_res_{i} = requant_{index} + (requant_{index + 1} << 16);"

        if offset == 1:
            yield f"((short*) output)[1] = (short) requant_0;"

        for i in range(num_halfwords):
            yield f"output[{offset + i}] = packed_res_{i};"

        if (offset + num_sums) % 2 == 1:
            yield f"((short*) output)[{num_halfwords * 2}] = (short) requant_{num_halfwords * 2};"


def tensordot_int16_impl(
    split_size: int,
    dimensions: Tuple[int, int, int],
    offsets: Tuple[int, int, int],
    x_strides: Tuple[int, int],
) -> str:
    """Code for a specialized version of tensordot, which computes `split_size` tensordot operations
    at the same time. Only works with `int16`. The generated function takes as input pointers to the
    output, tensor, and kernel, which must be word-aligned. However, the stride can be half a word.
    """
    function_name = get_c_function_name(split_size, dimensions, offsets, x_strides)
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
    requantize_lines = _requantize_sums(split_size)
    write_out_lines = _write_sums_to_memory(split_size, output_offset, out_stride)

    def insert_lines(lines):
        return ("\n" + " " * 10).join(lines)

    # __WEAK allows multiple copies of the function to overwrite themselves, saving flash
    code = textwrap.dedent(
        f"""
        #include <arm_nnsupportfunctions.h>
        __STATIC_FORCEINLINE __WEAK int {function_name}(
            int *output, int *tensor, int *kernel, int *bias, int *requant_scale
        ) {{
          {_init_biased_accumulators(split_size)}

          {insert_lines(load_tensor_lines)}

          {insert_lines(load_kernel_lines)}

          {insert_lines(multiply_acc_lines)}

          {insert_lines(requantize_lines)}

          {insert_lines(write_out_lines)}
          return 0;
        }}
        """
    )
    print(code)
    return code
