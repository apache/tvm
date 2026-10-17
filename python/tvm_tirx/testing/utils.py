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

# pylint: disable=invalid-name,unnecessary-comprehension,redefined-outer-name
"""Numerical assertions, device availability, and TIRX test utilities."""

import inspect
import sys
import time

import ml_dtypes
import numpy as np
import pytest

import tvm_tirx

from . import env


def assert_allclose(actual, desired, rtol=1e-7, atol=1e-7, verbose=True):
    """Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `actual` and `desired` are not interchangeable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """
    actual = np.asanyarray(actual)
    desired = np.asanyarray(desired)
    np.testing.assert_allclose(actual.shape, desired.shape)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=verbose)


def device_enabled(target):
    """Return whether the target's runtime and local device are available."""
    if isinstance(target, dict):
        kind = target["kind"]
    elif hasattr(target, "kind"):
        kind = target.kind.name
    else:
        assert isinstance(target, str), "device_enabled requires a target as a string"
        kind = target.split()[0]
    if not env._runtime_enabled(kind):
        return False
    try:
        return bool(tvm_tirx.device_from_target(target).exist)
    except Exception:  # pylint: disable=broad-except
        return False


def identity_after(x, sleep):
    """Testing function to return identity after sleep

    Parameters
    ----------
    x : int
        The input value.

    sleep : float
        The amount of time to sleep

    Returns
    -------
    x : object
        The original value
    """
    if sleep:
        time.sleep(sleep)
    return x


def terminate_self():
    """Testing function to terminate the process."""
    sys.exit(-1)


def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file, *sys.argv[1:]]))


ml_dtypes_dict = {
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "bfloat16": ml_dtypes.bfloat16,
    "int4": ml_dtypes.int4,
}


def np_dtype_from_str(dtype: str) -> np.dtype:
    """Convert a string dtype to a numpy dtype."""
    return np.dtype(ml_dtypes_dict[dtype]) if dtype in ml_dtypes_dict else np.dtype(dtype)


def generate_random_array(dtype: str, shape: tuple) -> np.ndarray:
    """
    Generate a random array by generating random bits and casting to the target dtype.

    Supported dtypes:
      - "int8", "uint8", "float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"
    """
    try:
        np_dtype = np_dtype_from_str(dtype)

    except TypeError:
        raise ValueError("Provided dtype is not a valid numpy dtype.")

    # Determine the bit length for this dtype.
    bit_length = np_dtype.itemsize * 8

    # Choose an appropriate unsigned container type.
    if bit_length <= 8:
        container = np.uint8
    elif bit_length <= 16:
        container = np.uint16
    elif bit_length <= 32:
        container = np.uint32
    elif bit_length <= 64:
        container = np.uint64
    else:
        raise ValueError(f"Unsupported dtype bit length: {bit_length}")

    # Generate random integers in the full range of the bit length.
    random_ints = np.random.randint(0, 2**bit_length, size=shape, dtype=container)
    # Reinterpret the bit pattern as the desired dtype.
    res = random_ints.view(np_dtype)
    with np.errstate(invalid="ignore"):
        invalid_indices = np.where(~np.isfinite(res))
    for idx in zip(*invalid_indices):
        while True:
            with np.errstate(invalid="ignore"):
                if np.isfinite(res[idx]):
                    break
            # Generate a new random value for this specific position
            new_random_int = np.random.randint(0, 2**bit_length, size=1, dtype=container)
            res[idx] = new_random_int.view(np_dtype)[0]
    return res
