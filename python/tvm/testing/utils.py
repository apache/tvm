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
"""TVM testing utilities

Organization
************

This file contains functions expected to be called directly by a user
while writing unit tests.  Integrations with the pytest framework
for TVM's own test suite are in ``tests/python/conftest.py``.

Testing Markers
***************

We use pytest markers to specify the requirements of test functions.
Currently there is a single distinction that matters for our testing
environment: does the test require a gpu.  Tests that require a gpu are
tagged with the ``gpu`` pytest marker -- the only registered marker (see
the ``markers`` entry in ``pyproject.toml``).  This lets us select the
gpu subset of tests with ``pytest -m gpu`` (and exclude them on cpu-only
nodes with ``pytest -m "not gpu"``).

The ``gpu`` marker only controls which testing node a test runs on; it
does not check whether the required hardware or libraries are actually
present.  To gate a test on a specific capability, combine the marker
with a ``skipif`` that consults the memoized environment probes in
:py:mod:`tvm.testing.env`:

.. code-block:: python

    @pytest.mark.gpu
    @pytest.mark.skipif(not tvm.testing.env.has_cuda(), reason="need cuda")
    def test_cuda_vectorize_add():
        ...

There is one ``has_*`` (or ``is_*``) probe per capability -- for example
:py:func:`tvm.testing.env.has_gpu`, :py:func:`tvm.testing.env.has_cuda`,
and :py:func:`tvm.testing.env.has_vulkan`.  For optional Python packages,
prefer ``pytest.importorskip("pkg_name")`` instead of a ``skipif``.

To run a test against a variety of targets, parametrize over ``target`` with
``@pytest.mark.parametrize("target", [...])`` -- tag GPU targets with
``pytest.mark.gpu`` so the CI routes them to GPU nodes, and skip an unavailable
target with ``pytest.mark.skipif(not tvm.testing.device_enabled(target))``.  The
set of enabled targets is controlled by the ``TVM_TEST_TARGETS`` environment
variable, so the CI can run different targets on different testing nodes.

"""

import inspect
import os
import sys
import time

import ml_dtypes
import numpy as np
import pytest

import tvm

from . import env

SKIP_SLOW_TESTS = os.getenv("SKIP_SLOW_TESTS", "").lower() in {"true", "1", "yes"}


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
    """Check if a target should be used when testing.

    Gate a device-specific test on this with
    ``@pytest.mark.skipif(not tvm.testing.device_enabled(target))``.

    This allows the user to control which devices they are testing against. In
    tests, this should be used to check if a device should be used when said
    device is an optional part of the test.

    Parameters
    ----------
    target : str or Dict[str, Any] or tvm.target.Target
        Target string to check against

    Returns
    -------
    bool
        Whether or not the device associated with this target is enabled.

    Example
    -------
    >>> @pytest.mark.gpu
    >>> def test_mytest():
    >>>     for target in ["cuda", "llvm"]:
    >>>         if device_enabled(target):
    >>>             test_body...

    Here, `test_body` will only be reached by with `target="cuda"` on gpu test
    nodes and `target="llvm"` on cpu test nodes.
    """
    if isinstance(target, dict):
        target_kind = target["kind"]
    elif hasattr(target, "kind"):
        target_kind = target.kind.name
    else:
        assert isinstance(target, str), "device_enabled requires a target as a string"
        # Target strings may include extra flags; only compare the kind.
        target_kind = target.split(" ")[0]
    # Check only explicitly selected candidates of the requested kind.  In
    # particular, an unavailable selection must not silently enable LLVM.
    for candidate in env._tvm_test_targets():
        kind = candidate["kind"] if isinstance(candidate, dict) else candidate.split()[0]
        if kind != target_kind:
            continue
        if kind == "hexagon":
            return env.build_flag_enabled("USE_HEXAGON") and "ANDROID_SERIAL_NUMBER" in os.environ
        if not env._runtime_enabled(kind):
            continue
        if kind == "cuda" and "cudnn" in tvm.target.Target(candidate).attrs.get("libs", []):
            exists = tvm.get_global_func("tvm.contrib.cudnn.exists", allow_missing=True)
            if env.has_cudnn() and exists is not None and exists():
                return True
        elif tvm.device_from_target(candidate).exist:
            return True
    return False


slow = pytest.mark.skipif(
    SKIP_SLOW_TESTS,
    reason="Skipping slow test since the SKIP_SLOW_TESTS environment variable is 'true'",
)


def parameter(*values, ids=None, by_dict=None):
    """Convenience function to define pytest parametrized fixtures.

    Declaring a variable using ``tvm.testing.parameter`` will define a
    parametrized pytest fixture that can be used by test
    functions. This is intended for cases that have no setup cost,
    such as strings, integers, tuples, etc.  For cases that have a
    significant setup cost, use an ordinary pytest fixture
    instead.

    If a test function accepts multiple parameters defined using
    ``tvm.testing.parameter``, then the test will be run using every
    combination of those parameters.

    The parameter definition applies to all tests in a module.  If a
    specific test should have different values for the parameter, that
    test should be marked with ``@pytest.mark.parametrize``.

    Parameters
    ----------
    values : Any

       A list of parameter values.  A unit test that accepts this
       parameter as an argument will be run once for each parameter
       given.

    ids : List[str], optional

       A list of names for the parameters.  If None, pytest will
       generate a name from the value.  These generated names may not
       be readable/useful for composite types such as tuples.

    by_dict : Dict[str, Any]

       A mapping from parameter name to parameter value, to set both the
       values and ids.

    Returns
    -------
    function
       A function output from pytest.fixture.

    Example
    -------
    >>> size = tvm.testing.parameter(1, 10, 100)
    >>> def test_using_size(size):
    >>>     ... # Test code here

    Or

    >>> shape = tvm.testing.parameter((5,10), (512,1024), ids=['small','large'])
    >>> def test_using_size(shape):
    >>>     ... # Test code here

    Or

    >>> shape = tvm.testing.parameter(by_dict={'small': (5,10), 'large': (512,1024)})
    >>> def test_using_size(shape):
    >>>     ... # Test code here

    """

    if by_dict is not None:
        if values or ids:
            raise RuntimeError(
                "Use of the by_dict parameter cannot be used alongside positional arguments"
            )

        ids, values = zip(*by_dict.items())

    # Optional cls parameter in case a parameter is defined inside a
    # class scope.
    @pytest.fixture(params=values, ids=ids, scope="session")
    def as_fixture(*_cls, request):
        return request.param

    return as_fixture


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
