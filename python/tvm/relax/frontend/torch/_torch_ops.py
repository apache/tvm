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
# ruff: noqa: E731

"""Tensor expressions for PyTorch operations without an equivalent Relax operator."""

from tvm import arith, te, tirx


def avg_pool_divisor(data, ndim, kernel, stride, padding, ceil_mode, divisor):
    """Sum valid window elements and divide by the caller's fixed divisor."""

    def expand(value):
        values = (value,) if isinstance(value, int) else tuple(value)
        return values * ndim if len(values) == 1 else values

    kernel = expand(kernel)
    stride = kernel if stride is None or stride == [] else expand(stride)
    padding = expand(padding)
    spatial = list(data.shape)[-ndim:]
    output = []
    for length, k, s, p in zip(spatial, kernel, stride, padding):
        extent = (length + 2 * p - k + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode:
            extent = tirx.min(extent, (length + p + s - 1) // s)
        output.append(extent)
    shape = list(data.shape)[:-ndim] + output
    axes = [te.reduce_axis((0, k), f"r{i}") for i, k in enumerate(kernel)]
    dtype = "float32" if data.dtype in ("float16", "bfloat16") else data.dtype

    def window(*indices):
        positions = [indices[-ndim + i] * stride[i] - padding[i] + axes[i] for i in range(ndim)]
        valid = tirx.all(
            *[tirx.all(pos >= 0, pos < length) for pos, length in zip(positions, spatial)]
        )
        value = tirx.if_then_else(
            valid, data[(*indices[:-ndim], *positions)].astype(dtype), tirx.const(0, dtype)
        )
        return te.sum(value, axis=axes)

    sums = te.compute(shape, window, name="pool_sum")
    return te.compute(
        shape,
        lambda *i: (sums[i] / tirx.const(divisor, dtype)).astype(data.dtype),
        name="pool_divisor",
    )


def resize2d_antialias(data, size, scales, align_corners, method):
    """Separable, scale-widened filters with normalized, clipped boundary weights."""
    uint8 = data.dtype == "uint8"
    dtype = "float64" if data.dtype == "float64" or uint8 else "float32"
    if data.dtype not in ("float16", "bfloat16", "float32", "float64", "uint8"):
        raise ValueError("Antialiased resize requires floating-point or uint8 input")
    cubic = method == "bicubic"
    # Match PyTorch's horizontal-then-vertical accumulation order.
    for axis in (3, 2):
        length, output_length = data.shape[axis], size[axis - 2]
        if arith.Analyzer().can_prove_equal(length, output_length):
            continue
        if align_corners:
            scale = tirx.if_then_else(
                output_length > 1,
                (length - 1).astype(dtype) / tirx.Cast(dtype, output_length - 1),
                tirx.const(0, dtype),
            )
        elif scales[axis - 2] is not None:
            scale = tirx.const(1 / scales[axis - 2], dtype)
        else:
            scale = length.astype(dtype) / tirx.Cast(dtype, output_length)
        width = tirx.max(scale, tirx.const(1, dtype))
        support = width * (2 if cubic else 1)
        taps = arith.Analyzer().simplify(tirx.Cast("int64", tirx.ceil(support)) * 2 + 1)
        center = lambda i: scale * (tirx.Cast(dtype, i) + 0.5)
        start = lambda i: tirx.max(tirx.Cast("int64", center(i) - support + 0.5), 0)

        def weight(i, k):
            distance = tirx.abs((tirx.Cast(dtype, start(i) + k) - center(i) + 0.5) / width)
            if cubic:
                value = tirx.if_then_else(
                    distance < 1,
                    ((1.5 * distance - 2.5) * distance) * distance + 1,
                    tirx.if_then_else(
                        distance < 2, ((-0.5 * distance + 2.5) * distance - 4) * distance + 2, 0
                    ),
                )
            else:
                value = tirx.max(1 - distance, tirx.const(0, dtype))
            return tirx.if_then_else(start(i) + k < length, value, tirx.const(0, dtype))

        weights = te.compute((output_length, taps), weight, name=f"weights_{axis}")
        r = te.reduce_axis((0, taps), "tap")
        totals = te.compute(
            (output_length,), lambda i: te.sum(weights[i, r], axis=r), name="weight_sum"
        )
        normalized = te.compute(
            (output_length, taps), lambda i, k: weights[i, k] / totals[i], name="normalized_weights"
        )
        if uint8:
            # PyTorch rounds uint8 pixels after each pass using int16 filter coefficients.
            y, k = te.reduce_axis((0, output_length), "y"), te.reduce_axis((0, taps), "k")
            maximum = te.compute(
                (), lambda: te.max(normalized[y, k], axis=[y, k]), name="max_weight"
            )
            precision = tirx.const(22, "int32")
            for bits in reversed(range(22)):
                precision = tirx.if_then_else(
                    maximum[()] * (1 << (bits + 1)) + 0.5 >= (1 << 15), bits, precision
                )
            shift = te.compute((), lambda: precision, name="weight_precision")

            def quantize(i, k):
                value = normalized[i, k] * (1 << shift[()])
                return tirx.Cast("int32", value + tirx.if_then_else(value < 0, -0.5, 0.5))

            coefficients = te.compute((output_length, taps), quantize, name="integer_weights")
        else:
            coefficients = normalized
        shape = list(data.shape)
        shape[axis] = output_length

        def resample(*indices):
            source = list(indices)
            position = start(indices[axis]) + r
            source[axis] = tirx.min(position, length - 1)
            stop = tirx.min(tirx.Cast("int64", center(indices[axis]) + support + 0.5), length)
            acc_dtype = "int32" if uint8 else dtype
            return te.sum(
                tirx.if_then_else(
                    position < stop,
                    data[tuple(source)].astype(acc_dtype) * coefficients[indices[axis], r],
                    tirx.const(0, acc_dtype),
                ),
                axis=r,
            )

        result = te.compute(shape, resample, name=f"resample_{axis}")

        def convert(*i):
            value = result[i]
            if uint8:
                value = (value + (1 << (shift[()] - 1))) >> shift[()]
                value = tirx.min(tirx.max(value, 0), 255)
            source = list(i)
            source[axis] = tirx.min(source[axis], length - 1)
            return tirx.if_then_else(
                length == output_length, data[tuple(source)], value.astype(data.dtype)
            )

        data = te.compute(shape, convert, name="resized")
    return data
