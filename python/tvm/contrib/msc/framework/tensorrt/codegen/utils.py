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
"""tvm.contrib.msc.framework.tensorrt.codegen.utils"""

import io
import struct
import numpy as np


def enum_dtype(array: np.ndarray) -> int:
    """Get TensorRT DType enum from array.

    Parameters
    ----------
    array: np.ndarray
        The source array.

    Returns
    -------
    dtype: int
        The dtype enum.
    """

    if array.dtype == np.float32:
        return 0
    if array.dtype == np.float16:
        return 1
    if array.dtype == np.int8:
        return 2
    if array.dtype == np.int32:
        return 3
    raise Exception("Unexpected dtype {}, no matching tensorrt dtype".format(array.dtype))


def float_to_hex(value: float) -> str:
    """Change float to hex.

    Parameters
    ----------
    value: float
        The float value.

    Returns
    -------
    hex: str
        The hex format string.
    """

    return hex(struct.unpack("<I", struct.pack("<f", value))[0])


def array_to_hex(array: np.ndarray) -> str:
    """Change array to hex.

    Parameters
    ----------
    array: np.ndarray
        The source array.

    Returns
    -------
    hex: str
        The hex format string.
    """

    return " ".join([float_to_hex(float(f))[2:] for f in array.flatten()])


def write_weight(name: str, weight: np.ndarray, f_handler: io.TextIOWrapper):
    """Write array to file in TensorRT format.

    Parameters
    ----------
    name: str
        The array name
    weight: np.ndarray
        The weight data.
    f_handler: io.TextIOWrapper
        The file handler
    """

    f_handler.write(
        "{} {} {} {}\n".format(name, enum_dtype(weight), weight.size, array_to_hex(weight))
    )
