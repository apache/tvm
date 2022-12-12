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

# pylint: disable=invalid-name


"""Common hexagon specific utilities"""
import math
import struct
from typing import Tuple
from tvm import te


def n11c_1024c_2d(n, h, w, c):
    """Return index map for n11c_1024 2d layout"""
    return [n, h, w, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def n11c_1024c_1d(n, h, w, c):
    """Return index map for n11c_1024 1d layout"""
    return [n, h, w, c // 1024, c % 1024]


def nhwc_8h2w32c2w_2d(n, h, w, c):
    """Return index map for nhwc_8h2w32c2w 2d layout"""
    return [n, h // 8, w // 4, c // 32, te.AXIS_SEPARATOR, h % 8, (w % 4) // 2, c % 32, w % 2]


def nhwc_8h2w32c2w_1d(n, h, w, c):
    """Return index map for nhwc_8h2w32c2w 1d layout"""
    return [n, h // 8, w // 4, c // 32, h % 8, (w % 4) // 2, c % 32, w % 2]


def nhw_32h16w_2d(n, h, w):
    """Return index map for nhw_32h16w 2d layout"""
    return [n, h // 32, w // 16, te.AXIS_SEPARATOR, h % 32, w % 16]


def nhwc_4h4w32c_1d(n, h, w, c):
    """Return index map for nhwc_4h4232c 1d layout"""
    return [n, h // 4, w // 4, c // 32, h % 4, w % 4, c % 32]


def nhwc_4h4w32c_2d(n, h, w, c):
    """Return index map for nhwc_4h4w32c 2d layout"""
    return [n, h // 4, w // 4, c // 32, te.AXIS_SEPARATOR, h % 4, w % 4, c % 32]


def nc_512c_1d(n, c):
    """Return index map for nc_512c 1d layout"""
    return [n, c // 512, c % 512]


def nc_512c_2d(n, c):
    """Return index map for nc_512c 2d layout"""
    return [n, c // 512, te.AXIS_SEPARATOR, c % 512]


def nc_1024c_2d(n, c):
    """Return index map for nc_1024c 2d layout"""
    return [n, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def nhwc_4h2w32c2w_2d(n, h, w, c):
    """Return index map for nhwc_4h2w32c2w 2d layout"""
    return [n, h // 4, w // 4, c // 32, te.AXIS_SEPARATOR, h % 4, (w % 4) // 2, c % 32, w % 2]


def nhwc_1024c_2d(n, h, w, c):
    """Return index map for nhwc_1024 2d layout"""
    return [n, h, w, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def nc_1024_2d(n, c):
    """Return index map for nc_1024 2d layout"""
    return [n, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def nhwc_2048c_2d(n, h, w, c):
    """Return index map for nhwc_2048 2d layout"""
    return [n, h, w, c // 2048, te.AXIS_SEPARATOR, c % 2048]


def nc_2048_2d(n, c):
    """Return index map for nc_2048 2d layout"""
    return [n, c // 2048, te.AXIS_SEPARATOR, c % 2048]


def nc_2048c_2d(n, c):
    """Return index map for nc_2048 2d layout"""
    return [n, c // 2048, te.AXIS_SEPARATOR, c % 2048]


def nhwc_8h8w32c_2d(n, h, w, c):
    """Return index map for nhwc_8h8w32c 2d layout"""
    return [n, h // 8, w // 8, c // 32, te.AXIS_SEPARATOR, h % 8, w % 8, c % 32]


def n11c_2048c_2d(n, h, w, c):
    """Return index map for n11c_2048c 2d layout"""
    return [n, h, w, c // 2048, te.AXIS_SEPARATOR, c % 2048]


def iohw_16i32o2i_1d(height, width, in_channel, out_channel):
    return [
        in_channel // 32,
        out_channel // 32,
        height,
        width,
        (in_channel % 32) // 2,
        out_channel % 32,
        in_channel % 2,
    ]


def ohwi32o_1d(height, width, in_channel, out_channel):
    return [out_channel // 32, height, width, in_channel, out_channel % 32]


def ncw_32c64w_2d(n, c, w):
    """Return index map for ncw_32c64w 2d layout"""
    return [n, c // 32, w // 64, te.AXIS_SEPARATOR, c % 32, w % 64]


def get_layout_transform_fn(layout):
    """Return index map function as per the layout string"""
    if layout == "nhwc-8h2w32c2w-2d":
        return nhwc_8h2w32c2w_2d
    if layout == "nhwc-8h2w32c2w-1d":
        return nhwc_8h2w32c2w_1d
    if layout == "n11c-1024c-2d":
        return n11c_1024c_2d
    if layout == "n11c-1024c-1d":
        return n11c_1024c_1d
    if layout == "nhwc-1024c-2d":
        return nhwc_1024c_2d
    if layout == "nc-1024-2d":
        return nc_1024_2d
    if layout == "nhw-32h16w-2d":
        return nhw_32h16w_2d
    if layout == "nhwc-4h4w32c-2d":
        return nhwc_4h4w32c_2d
    if layout == "nhwc-4h4w32c-1d":
        return nhwc_4h4w32c_1d
    if layout == "nc-512c-2d":
        return nc_512c_2d
    if layout == "nc-512c-1d":
        return nc_512c_1d
    if layout == "nhwc-4h2w32c2w-2d":
        return nhwc_4h2w32c2w_2d
    if layout == "nc-1024c-2d":
        return nc_1024c_2d
    if layout == "iohw-16i32o2i-1d":
        return iohw_16i32o2i_1d
    if layout == "nhwc-2048c-2d":
        return nhwc_2048c_2d
    if layout == "nc-2048-2d":
        return nc_2048_2d
    if layout == "nc-2048c-2d":
        return nc_2048c_2d
    if layout == "nhwc-8h8w32c-2d":
        return nhwc_8h8w32c_2d
    if layout == "n11c-2048c-2d":
        return n11c_2048c_2d
    if layout == "ohwi32o-1d":
        return ohwi32o_1d
    if layout == "ncw-32c64w-2d":
        return ncw_32c64w_2d
    raise RuntimeError(f"Unexpected layout '{layout}'")


def get_fixed_point_value(flp: float, dtype: str = "int16") -> Tuple[int, int]:
    """
    Return fixed-point value and the corresponding log2 of the scale factor used to compute
    this value.

    Parameters
    ----------
    flp : float
        Floating-point value to be converted
    dtype : str
        Type of the resulting fixed-point value. By default, it's set to "int16"

    Returns
    -------
    fixed_point_value : int
        Fixed-point value for the given floating-point value
    exp_scale_factor : int
        log2 of the scale factor

    Convert floating-point value into fixed-point number. This is done by
    multiplying the value by a scaling factor and then rounding it to the nearest
    integer value.

    As per IEEE-754 standard, a floating-point value can be represented as follows
    [see: https://en.wikipedia.org/wiki/IEEE_754-1985]:
        (-1)^S * M * 2^(E-Bias)

    Here,
    * S is the signed bit (0 or 1).
    * M is the mantissa. It's composed of an implicit 1 for the normalized floating-point
      values or 0 for the denormalized values, and the fraction part. This ensures that
      mantissa is always within [0, 2) range. Please note that this function doesn't
      handle denormalized values.
    * E is the exponent.

    In single precision, 23 bits are used to represent the fraction part of
    the mantissa (and therefore, '23' shows up in one of the computations below) and
    8 bits are used for the exponent. Since exponent field needs to reperesent both
    positive and negative values, a bias (127 for single precision) is added to the actual
    value. Therefore, to compute the actual exponent, 127 must be subtracted from the stored
    value.

    As mentioned above, to find the corresponding fixed-point number, we multiply the
    value with a scaling factor and then round it to the nearest integer. The scaling factor
    is chosen to be a power for 2 and it's the largest value that can be safely multiplied
    to the floating-point value, without causing the resulting value to overflow the range
    of the integer type used to represent the fixed-point value.

    So, if we assume the scaling factor to be 2^x, the resulting fixed-point value will be:
        round((-1)^S * (M) * 2^(E-Bias) * 2^x)

    This can be simplified to:
        round((-1)^S * M * 2^(E-Bias+x)

    Now, if 'int16' is used for fixed-point value, then it has to be >= -(2 * 2^14)
    and <= (2 * 2^14) - 1. Since M (Mantissa) is always < 2, in order for the fixed-point value
    to be within this range, 2^(E - Bias + x) must be <= 2^14 - 1.
    And, if we ignore -1, (E - Bias + x) should be <= 14. Note: if mantissa gets too close to 2,
    this will cause the resulting value to go out of range and require it to be saturated.
    In the following implementation, we perform range check and adjust the scale to avoid
    saturation.
    For most cases, 2^x, where x = 14 - (E - Bias) or 14 - (E - 127) for single precision, is the
    best scaling factor for 'int16' type that can be used to convert the floating-point value to
    fixed-point with the least amount of precision loss.


    Here is a more rigorous explanation of the above, for non-negative scale values, which are of
    interest. M < 2, so M * 2^(E-Bias+x) < 2 ^ (E-Bias+x+1)   [Note: LHS is a fraction, RHS int]
    => round(M * 2^(E-Bias+x)) <= 2 ^ (E-Bias+x+1)  [Note the "<=", not "<"]
    We want x s.t. round(M * 2^(E-Bias+x)) <= 2^15 - 1
    We know round(M * 2^(E-Bias+x)) <= 2^(E-Bias+x+1)
    It will be sufficient to choose x s.t. 2^(E-Bias+x+1) <= 2^15 - 1
    That is, max x. s.t. 2^(E-Bias+x+1) < 2^15
    E-Bias+x+1 < 15
    E-Bias+x+1 <= 14
    Max x will make E-Bias+x+1 = 14
    x = 13 - E + Bias

    Additonal notes on various floating-point values:
    ------------------------------------------------
    1) Denormalized values: causes assertion failure. The problem with the denormalized values
        is that they require a very large scale factor (>= 2^127) to be converted to a fixed-point
        value. As the denormalzied values get smaller, the scale factor becomes too large to be
        represented as a IEEE-754 floating point value (as being done in the computaton below)
        and therefore, the denormalized values aren't being handled here.
    2) NaN and INF: assertion failure
    """

    def within_range(val, dtype):
        if dtype == "int16":
            return -32768 <= val <= 32767
        raise RuntimeError(f"Unsupported dtype, {dtype}'")

    # Make sure that 'flp' isn't NaN or infinity
    if math.isnan(flp) or math.isinf(flp):
        raise RuntimeError("NaN or INF can not be represented as fixed-point")

    flp_f = struct.pack("f", flp)
    flp_i = struct.unpack("I", flp_f)
    exp_stored_value = (flp_i[0] >> 23) & 0xFF

    if exp_stored_value == 0:
        raise RuntimeError(
            "Denormalized values are not considered for float -> fixed-point conversion!"
        )

    exp_value = ((flp_i[0] >> 23) & 0xFF) - 127
    if dtype == "int16":
        max_bits = 14
    else:
        raise RuntimeError(f"Unsupported dtype, {dtype}'")

    exp_scale_factor = max_bits - exp_value  # log2 of the scale_factor

    if exp_scale_factor > 127:
        raise RuntimeError("Value too small for fixed-point conversion!")

    # Scaling factor = 2^exp_scale_factor
    # Since exp_scale_factor can be -ve or +ve, scaling factor is calculated by first
    # representing the value in the binary format as per IEEE floating-point standand and then
    # reinterpreting it as a float using struct.pack and struct.unpack functions.
    # struct.pack returns a bytes object packed as integer and struct.unpack
    # unpacks this bytes object into float.
    scale = ((exp_scale_factor + 127) & 0xFF) << 23
    scale_i = struct.pack("I", scale)
    scale_f = struct.unpack("f", scale_i)
    fixed_point_value = int(round(flp * scale_f[0]))

    if not within_range(fixed_point_value, dtype):
        # Adjust scale factor to avoid overflow.
        exp_scale_factor -= 1
        scale = ((exp_scale_factor + 127) & 0xFF) << 23
        scale_i = struct.pack("I", scale)
        scale_f = struct.unpack("f", scale_i)
        fixed_point_value = int(round(flp * scale_f[0]))

    return fixed_point_value, exp_scale_factor


def saturate(x: te.Tensor, dtype: str):
    """Saturate value for the specified data type"""
    return te.max(te.min_value(dtype), te.min(x, te.max_value(dtype)))
