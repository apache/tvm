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
"""
Helper utility Enums and Functions used through out code generation.

The rest of the utility functions are misc.
Refer to the description inside such functions
"""

from inspect import signature
from enum import Enum
from typing import Union, Tuple
import numpy as np  # type: ignore

import tvm  # type: ignore
from tvm import relay


class QConv2DArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.conv2d arguments.
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class RequantArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.requantize arguments.
    """

    IFM_SCALE = 1
    IFM_ZERO_POINT = 2
    OFM_SCALE = 3
    OFM_ZERO_POINT = 4


class BiasAddArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of qnn.bias_add arguments.
    """

    BIASES = 1


class ClipArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of clip arguments.
    """

    A_MIN = 1
    A_MAX = 2


class BinaryElementwiseArgs(Enum):
    """This is a helper enums to access the correct index
    of binary elementwise arguments
    """

    ifm = 0
    ifm2 = 1
    ifm_scale = 2
    ifm_zero_point = 3
    ifm2_scale = 4
    ifm2_zero_point = 5
    ofm_scale = 6
    ofm_zero_point = 7


def is_composite_func(func: relay.Function, name: str) -> bool:
    """
    This method checks whether the call is to
    a composite function of a given name.

    Parameters
    ----------
    func : relay.Function
        The header to be displayed along with the dump.

    name : str
        The candidate name to be checked

    Returns
    --------
    a boolean
    """

    if not hasattr(func, "attrs"):
        return False
    if "Composite" not in func.attrs.keys():
        return False
    composite_name = func.attrs["Composite"]

    return composite_name == name


def get_range_for_dtype_str(dtype: str) -> Tuple[int, int]:
    """
    Produce the min,max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """

    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info.min, type_info.max


def round_away_zero(f: Union[float, np.double, np.single, np.float32, np.float64]) -> np.float64:
    """Round the number away from zero towards +inf / -inf"""
    offset = -0.5 if (f < 0) else 0.5
    return np.trunc(f + offset)


def round_up(a: int, b: int) -> int:
    """Round up to a multiple of b"""
    return ((a + b - 1) // b) * b


def get_accelerator_config():
    """Get the variant of the accelerator to compile for"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethosu.get_compiler_attrs")()
    return compiler_attrs.accelerator_config


def get_arg_count(func):
    """Helper function to get the number of
    arguments in a python function"""
    sig = signature(func)
    return len(sig.parameters)


def get_dim_value(layout: str, dim: int):
    """This is a helper function to retrieve the value
    of the dimension given the shape and the layout
    """
    assert isinstance(layout, str)
    assert dim in list(layout)
    for idx, dim_char in enumerate(layout):
        if dim_char == dim:
            return idx
    return None


def calculate_size_bytes(expr):
    """This is a helper function to calculate the number
    of bytes required to hold the tensor/relay.expr"""
    try:
        type_info = np.iinfo(expr.checked_type.dtype)
    except ValueError:
        type_info = np.finfo(expr.checked_type.dtype)
    element_size = type_info.bits // 8
    elements = np.prod(list(expr.checked_type.shape))
    return element_size * elements
