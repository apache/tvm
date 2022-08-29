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

""" Hexagon pytest utility functions """

from typing import List, Optional, Union
import collections
import numpy as np


def get_test_id(*test_params, test_param_descs: List[Optional[str]] = None) -> str:
    """
    An opinionated alternative to pytest's default algorithm for generating a
    test's ID string.  Intended to make it easier for human readers to
    interpret the test IDs.

    'test_params': The sequence of pytest parameter values supplied to some unit
       test.

    'test_param_descs': An (optional) means to provide additional text for some/all of the
       paramuments in 'test_params'.

       If provided, then len(test_params) must equal len(test_param_descs).
       Each element test_param_descs that is a non-empty string will be used
       in some sensible way in this function's returned string.
    """

    assert len(test_params) > 0

    if test_param_descs is None:
        test_param_descs = [None] * len(test_params)
    else:
        assert len(test_param_descs) == len(test_params)

    def get_single_param_chunk(param_val, param_desc: Optional[str]):
        if isinstance(param_val, list):
            # Like str(list), but avoid the whitespace padding.
            val_str = "[" + ",".join(str(x) for x in param_val) + "]"
            need_prefix_separator = False

        elif isinstance(param_val, bool):
            if param_val:
                val_str = "T"
            else:
                val_str = "F"
            need_prefix_separator = True

        elif isinstance(param_val, TensorContentConstant):
            val_str = f"const[{param_val.elem_value}]"
            need_prefix_separator = True

        elif isinstance(param_val, TensorContentDtypeMin):
            val_str = "min"
            need_prefix_separator = True

        elif isinstance(param_val, TensorContentDtypeMax):
            val_str = "max"
            need_prefix_separator = True

        elif isinstance(param_val, TensorContentRandom):
            val_str = "random"
            need_prefix_separator = True

        elif isinstance(param_val, TensorContentSequentialCOrder):
            val_str = f"seqC[start:{param_val.start_value},inc:{param_val.increment}]"
            need_prefix_separator = True

        else:
            val_str = str(param_val)
            need_prefix_separator = True

        if param_desc and need_prefix_separator:
            return f"{param_desc}:{val_str}"
        elif param_desc and not need_prefix_separator:
            return f"{param_desc}{val_str}"
        else:
            return val_str

    chunks = [
        get_single_param_chunk(param_val, param_desc)
        for param_val, param_desc in zip(test_params, test_param_descs)
    ]
    return "-".join(chunks)


def get_multitest_ids(
    multitest_params_list: List[List], param_descs: Optional[List[Optional[str]]]
) -> List[str]:
    """
    A convenience function for classes that use both 'tvm.testing.parameters' and 'get_test_id'.

    This function provides a workaround for a specific quirk in Python, where list-comprehension
    can't necessarily access the value of another class-variable, discused here:
    https://stackoverflow.com/q/13905741
    """
    return [
        get_test_id(*single_test_param_list, test_param_descs=param_descs)
        for single_test_param_list in multitest_params_list
    ]


def get_numpy_dtype_info(dtype) -> Union[np.finfo, np.iinfo]:
    """
    Return an appropriate 'np.iinfo' or 'np.finfo' object corresponding to
    the specified Numpy dtype.

    'dtype' must be a value that 'numpy.dtype(...)' can handle.
    """
    np_dtype = np.dtype(dtype)
    kind = np_dtype.kind

    if kind == "f":
        return np.finfo(np_dtype)
    elif kind == "i":
        return np.iinfo(np_dtype)
    else:
        raise TypeError(f"dtype ({dtype}) must indicate some floating-point or integral data type")


TensorContentConstant = collections.namedtuple("TensorContentConstant", ["elem_value"])
TensorContentSequentialCOrder = collections.namedtuple(
    "TensorContentSequentialCOrder", ["start_value", "increment"]
)
TensorContentRandom = collections.namedtuple("TensorContentRandom", [])
TensorContentDtypeMin = collections.namedtuple("TensorContentDtypeMin", [])
TensorContentDtypeMax = collections.namedtuple("TensorContentDtypeMax", [])


def create_populated_numpy_ndarray(
    input_shape: Union[list, tuple], dtype: str, input_tensor_populator
) -> np.ndarray:
    """
    Create a numpy tensor with the specified shape, dtype, and content.
    """
    itp = input_tensor_populator  # just for brevity

    if isinstance(itp, TensorContentConstant):
        return np.full(tuple(input_shape), itp.elem_value, dtype=dtype)

    elif isinstance(itp, TensorContentDtypeMin):
        info = get_numpy_dtype_info(dtype)
        return np.full(tuple(input_shape), info.min, dtype=dtype)

    elif isinstance(itp, TensorContentDtypeMax):
        info = get_numpy_dtype_info(dtype)
        return np.full(tuple(input_shape), info.max, dtype=dtype)

    elif isinstance(itp, TensorContentRandom):
        return np.random.random(input_shape).astype(dtype)

    elif isinstance(itp, TensorContentSequentialCOrder):
        a = np.empty(tuple(input_shape), dtype)

        with np.nditer(a, op_flags=["writeonly"], order="C") as iterator:
            next_elem_val = itp.start_value
            for elem in iterator:
                elem[...] = next_elem_val
                next_elem_val += itp.increment
        return a

    else:
        raise ValueError(f"Unexpected input_tensor_populator type: {type(itp)}")
