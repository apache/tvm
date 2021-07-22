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
Helper utility Enums and Functions used through out codegen

The enums are there to indicate which argument of each relay operator
corresponds with which input.
e.g., input zero point of qnn.conv2d is 4th argument(3rd index)

The rest of the utility functions are misc.
Refer to the description inside such functions
"""

from enum import Enum
import numpy as np

from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.backend.contrib.ethosu import preprocess


class QConv2DArgs(Enum):
    """
    This is a helper enums to access the correct index
    qnn conv2d arguments
    """

    ifm = 0
    weights = 1
    ifm_zero_point = 2
    weights_zero_point = 3
    ifm_scale = 4
    weights_scale = 5


class RequantArgs(Enum):
    """
    This is a helper enums to access the correct index
    qnn requantize arguments
    """

    ifm_scale = 1
    ifm_zero_point = 2
    ofm_scale = 3
    ofm_zero_point = 4


class BiasAddArgs(Enum):
    """
    This is a helper enums to access the correct index
    qnn bias_add arguments
    """

    biases = 1


class ClipArgs(Enum):
    """
    This is a helper enums to access the correct index
    qnn bias_add arguments
    """

    a_min = 1
    a_max = 2


class MaxPoolArgs(Enum):
    """
    This is a helper enums to access the correct index
    max pool arguments
    """

    ifm = 0


class AddArgs(Enum):
    """This is a helper enums to access the correct index
    max pool arguments
    """

    ifm0 = 0
    ifm1 = 1
    ifm0_scale = 2
    ifm0_zero_point = 3
    ifm1_scale = 4
    ifm1_zero_point = 5
    ofm_scale = 6
    ofm_zero_point = 7


def is_composite_func(func, name):
    """
    This a method to check whether the call is to
    a composite function of the "name".
    """
    if not hasattr(func, "attrs"):
        return False
    if "Composite" not in func.attrs.keys():
        return False
    composite_name = func.attrs["Composite"]

    if composite_name != name:
        return False
    return True


def get_range_for_dtype_str(dtype):
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


def round_away_zero(f):
    """round the number away from zero towards +inf / -inf"""
    offset = -0.5 if (f < 0) else 0.5
    return np.trunc(f + offset)


def round_up(a, b):
    """round up to a multiple of b"""
    return ((a + b - 1) // b) * b


# pylint: disable=unused-argument
def partition_for_ethosu(mod, params=None, **opts):
    """This helper function partition the relay graph as produced by the
    relay frontend for a given model into external functions
    to be presented to the codegen.

    Parameters
    ----------
    mod : IRModule
        The IRModule that gets generated from a relay frontend
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    mod : IRModule
        The partitioned IRModule with external global functions
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    pattern = relay.op.contrib.get_pattern_table("ethosu")
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.MergeComposite(pattern)(mod)
    mod = relay.transform.AnnotateTarget("ethosu")(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    mod = preprocess.preprocess_ext_io()(mod)
    return mod


def get_dim_value(layout, dim):
    """This is a helper function to retrieve the value
    of the dimension given the shape and the layout
    """
    assert isinstance(layout, str)
    assert dim in list(layout)
    for idx, dim_char in enumerate(layout):
        if dim_char == dim:
            return idx
    return None
