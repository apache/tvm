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
# ruff: noqa: E501

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

import copy
import copyreg
import ctypes
import functools
import inspect
import logging
import os
import pickle
import platform
import runpy
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import tvm
import tvm.arith
import tvm.support.utils
import tvm.te
import tvm.tirx
from tvm.contrib import cudnn
from tvm.support import nvcc

_TRUTHY_ENV_VALUES = frozenset({"true", "1", "yes"})


def _env_truthy(name: str, default: str = "") -> bool:
    """Return whether an environment variable is set to a truthy value."""
    return os.getenv(name, default).strip().lower() in _TRUTHY_ENV_VALUES


SKIP_SLOW_TESTS = _env_truthy("SKIP_SLOW_TESTS")
IS_IN_CI = os.getenv("CI", "").strip().lower() == "true"
_REQUEST_HOOK_INITIALIZERS = {}

skip_if_wheel_test = pytest.mark.skipif(
    _env_truthy("WHEEL_TEST"),
    reason="Test not supported in wheel.",
)


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


def check_numerical_grads(
    function, input_values, grad_values, function_value=None, delta=1e-3, atol=1e-2, rtol=0.1
):
    """A helper function that checks that numerical gradients of a function are
    equal to gradients computed in some different way (analytical gradients).

    Numerical gradients are computed using finite difference approximation. To
    reduce the number of function evaluations, the number of points used is
    gradually increased if the error value is too high (up to 5 points).

    Parameters
    ----------
    function
        A function that takes inputs either as positional or as keyword
        arguments (either `function(*input_values)` or `function(**input_values)`
        should be correct) and returns a scalar result. Should accept numpy
        ndarrays.

    input_values : Dict[str, numpy.ndarray] or List[numpy.ndarray]
        A list of values or a dict assigning values to variables. Represents the
        point at which gradients should be computed.

    grad_values : Dict[str, numpy.ndarray] or List[numpy.ndarray]
        Gradients computed using a different method.

    function_value : float, optional
        Should be equal to `function(**input_values)`.

    delta : float, optional
        A small number used for numerical computation of partial derivatives.
        The default 1e-3 is a good choice for float32.

    atol : float, optional
        Absolute tolerance. Gets multiplied by `sqrt(n)` where n is the size of a
        gradient.

    rtol : float, optional
        Relative tolerance.
    """
    # If input_values is a list then function accepts positional arguments
    # In this case transform it to a function taking kwargs of the form {"0": ..., "1": ...}
    if not isinstance(input_values, dict):
        input_len = len(input_values)
        input_values = {str(idx): val for idx, val in enumerate(input_values)}

        def _function(_input_len=input_len, _orig_function=function, **kwargs):
            return _orig_function(*(kwargs[str(i)] for i in range(input_len)))

        function = _function

        grad_values = {str(idx): val for idx, val in enumerate(grad_values)}

    if function_value is None:
        function_value = function(**input_values)

    # a helper to modify j-th element of val by a_delta
    def modify(val, j, a_delta):
        val = val.copy()
        val.reshape(-1)[j] = val.reshape(-1)[j] + a_delta
        return val

    # numerically compute a partial derivative with respect to j-th element of the var `name`
    def derivative(x_name, j, a_delta):
        modified_values = {
            n: modify(val, j, a_delta) if n == x_name else val for n, val in input_values.items()
        }
        return (function(**modified_values) - function_value) / a_delta

    def compare_derivative(j, n_der, grad):
        der = grad.reshape(-1)[j]
        return np.abs(n_der - der) < atol + rtol * np.abs(n_der)

    for x_name, grad in grad_values.items():
        if grad.shape != input_values[x_name].shape:
            raise AssertionError(
                f"Gradient wrt '{x_name}' has unexpected shape {grad.shape}, expected {input_values[x_name].shape} "
            )

        ngrad = np.zeros_like(grad)

        wrong_positions = []

        # compute partial derivatives for each position in this variable
        for j in range(np.prod(grad.shape)):
            # forward difference approximation
            nder = derivative(x_name, j, delta)

            # if the derivative is not equal to the analytical one, try to use more
            # precise and expensive methods
            if not compare_derivative(j, nder, grad):
                # central difference approximation
                nder = (derivative(x_name, j, -delta) + nder) / 2

                if not compare_derivative(j, nder, grad):
                    # central difference approximation using h = delta/2
                    cnder2 = (
                        derivative(x_name, j, delta / 2) + derivative(x_name, j, -delta / 2)
                    ) / 2
                    # five-point derivative
                    nder = (4 * cnder2 - nder) / 3

            # if the derivatives still don't match, add this position to the
            # list of wrong positions
            if not compare_derivative(j, nder, grad):
                wrong_positions.append(np.unravel_index(j, grad.shape))

            ngrad.reshape(-1)[j] = nder

        wrong_percentage = int(100 * len(wrong_positions) / np.prod(grad.shape))

        dist = np.sqrt(np.sum((ngrad - grad) ** 2))
        grad_norm = np.sqrt(np.sum(ngrad**2))

        if not (np.isfinite(dist) and np.isfinite(grad_norm)):
            raise ValueError(
                f"NaN or infinity detected during numerical gradient checking wrt '{x_name}'\n"
                f"analytical grad = {grad}\n numerical grad = {ngrad}\n"
            )

        # we multiply atol by this number to make it more universal for different sizes
        sqrt_n = np.sqrt(float(np.prod(grad.shape)))

        if dist > atol * sqrt_n + rtol * grad_norm:
            raise AssertionError(
                f"Analytical and numerical grads wrt '{x_name}' differ too much\n"
                f"analytical grad = {grad}\n numerical grad = {ngrad}\n"
                f"{wrong_percentage}% of elements differ, first 10 of wrong positions: {wrong_positions[:10]}\n"
                "distance > atol*sqrt(n) + rtol*grad_norm\n"
                f"distance {dist} > {atol}*{sqrt_n} + {rtol}*{grad_norm}"
            )

        max_diff = np.max(np.abs(ngrad - grad))
        avg_diff = np.mean(np.abs(ngrad - grad))
        logging.info(
            "Numerical grad test wrt '%s' of shape %s passes, "
            "dist = %f, max_diff = %f, avg_diff = %f",
            x_name,
            grad.shape,
            dist,
            max_diff,
            avg_diff,
        )


def assert_prim_expr_equal(lhs, rhs):
    """Assert lhs and rhs equals to each iother.

    Parameters
    ----------
    lhs : tvm.tirx.Expr
        The left operand.

    rhs : tvm.tirx.Expr
        The left operand.
    """
    ana = tvm.arith.Analyzer()
    if not ana.can_prove_equal(lhs, rhs):
        raise ValueError(f"{lhs} and {rhs} are not equal")


def check_bool_expr_is_true(bool_expr, vranges, cond=None):
    """Check that bool_expr holds given the condition cond
    for every value of free variables from vranges.

    For example, ``2x > 4y`` solves to ``x > 2y`` given ``x in (0, 10)``
    and ``y in (0, 10)``. Here bool_expr is ``x > 2y``,
    vranges is ``{x: (0, 10), y: (0, 10)}``, cond is ``2x > 4y``.
    We create iterations to check::

        for x in range(10):
            for y in range(10):
                assert !(2x > 4y) || (x > 2y)

    Parameters
    ----------
    bool_expr : tvm.ir.Expr
        Boolean expression to check
    vranges: Dict[tvm.tirx.expr.Var, tvm.ir.Range]
        Free variables and their ranges
    cond: tvm.ir.Expr
        extra conditions needs to be satisfied.
    """
    if cond is not None:
        bool_expr = tvm.te.any(tvm.tirx.Not(cond), bool_expr)

    def _run_expr(expr, vranges):
        """Evaluate expr for every value of free variables
        given by vranges and return the tensor of results.
        """

        def _compute_body(*us):
            vmap = {v: u + r.min for (v, r), u in zip(vranges.items(), us)}
            return tvm.tirx.stmt_functor.substitute(expr, vmap)

        A = tvm.te.compute([r.extent.value for v, r in vranges.items()], _compute_body)
        args = [tvm.runtime.empty(A.shape, A.dtype)]
        mod = tvm.compile(tvm.IRModule.from_expr(tvm.te.create_prim_func([A])))
        mod(*args)
        return args[0].numpy()

    res = _run_expr(bool_expr, vranges)
    if not np.all(res):
        indices = list(np.argwhere(res == 0)[0])
        counterex = [(str(v), i + r.min) for (v, r), i in zip(vranges.items(), indices)]
        counterex = sorted(counterex, key=lambda x: x[0])
        counterex = ", ".join([v + " = " + str(i) for v, i in counterex])
        ana = tvm.arith.Analyzer()
        raise AssertionError(
            f"Expression {ana.simplify(bool_expr)}\nis not true on {vranges}\n"
            f"Counterexample: {counterex}"
        )


def check_int_constraints_trans_consistency(constraints_trans, vranges=None):
    """Check IntConstraintsTransform is a bijective transformation.

    Parameters
    ----------
    constraints_trans : arith.IntConstraintsTransform
        Integer constraints transformation
    vranges: Dict[tvm.tirx.Var, tvm.ir.Range]
        Free variables and their ranges
    """
    if vranges is None:
        vranges = {}

    def _check_forward(constraints1, constraints2, varmap, backvarmap):
        ana = tvm.arith.Analyzer()
        all_vranges = vranges.copy()
        all_vranges.update({v: r for v, r in constraints1.ranges.items()})

        # Check that the transformation is injective
        cond_on_vars = tvm.tirx.const(1, "bool")
        for v in constraints1.variables:
            if v in varmap:
                # variable mapping is consistent
                v_back = ana.simplify(tvm.tirx.stmt_functor.substitute(varmap[v], backvarmap))
                cond_on_vars = tvm.te.all(cond_on_vars, v == v_back)
        # Also we have to check that the new relations are true when old relations are true
        cond_subst = tvm.tirx.stmt_functor.substitute(
            tvm.te.all(tvm.tirx.const(1, "bool"), *constraints2.relations), backvarmap
        )
        # We have to include relations from vranges too
        for v in constraints2.variables:
            if v in constraints2.ranges:
                r = constraints2.ranges[v]
                range_cond = tvm.te.all(v >= r.min, v < r.min + r.extent)
                range_cond = tvm.tirx.stmt_functor.substitute(range_cond, backvarmap)
                cond_subst = tvm.te.all(cond_subst, range_cond)
        cond_subst = ana.simplify(cond_subst)
        check_bool_expr_is_true(
            tvm.te.all(cond_subst, cond_on_vars),
            all_vranges,
            cond=tvm.te.all(tvm.tirx.const(1, "bool"), *constraints1.relations),
        )

    _check_forward(
        constraints_trans.src,
        constraints_trans.dst,
        constraints_trans.src_to_dst,
        constraints_trans.dst_to_src,
    )
    _check_forward(
        constraints_trans.dst,
        constraints_trans.src,
        constraints_trans.dst_to_src,
        constraints_trans.src_to_dst,
    )


def _get_targets(target_names=None):
    if target_names is None:
        target_names = _tvm_test_targets()

    if not target_names:
        target_names = DEFAULT_TEST_TARGETS

    targets = []
    for target in target_names:
        if isinstance(target, dict):
            target_kind = target["kind"]
        else:
            target_kind = target.split()[0]

        if target_kind == "cuda" and "cudnn" in tvm.target.Target(target).attrs.get("libs", []):
            is_enabled = tvm.support.libinfo().get("USE_CUDNN", "OFF").lower() in [
                "on",
                "true",
                "1",
            ]
            is_runnable = is_enabled and cudnn.exists()
        elif target_kind == "hexagon":
            is_enabled = tvm.support.libinfo().get("USE_HEXAGON", "OFF").lower() in [
                "on",
                "true",
                "1",
            ]
            # If Hexagon has compile-time support, we can always fall back
            is_runnable = is_enabled and "ANDROID_SERIAL_NUMBER" in os.environ
        else:
            is_enabled = tvm.runtime.enabled(target_kind)
            is_runnable = is_enabled and tvm.device_from_target(target).exist

        targets.append(
            {
                "target": target,
                "target_kind": target_kind,
                "is_enabled": is_enabled,
                "is_runnable": is_runnable,
            }
        )

    if all(not t["is_runnable"] for t in targets):
        if tvm.runtime.enabled("llvm"):
            logging.warning(
                "None of the following targets are supported by this build of TVM: %s."
                " Try setting TVM_TEST_TARGETS to a supported target. Defaulting to llvm.",
                target_names,
            )
            return _get_targets(["llvm"])

        raise RuntimeError(
            "None of the following targets are supported by this build of TVM: %s."
            " Try setting TVM_TEST_TARGETS to a supported target."
            " Cannot default to llvm, as it is not enabled." % target_names
        )

    return targets


DEFAULT_TEST_TARGETS = [
    "llvm",
    "cuda",
    "nvptx",
    {"kind": "vulkan", "from_device": 0},
    "opencl",
    {"kind": "opencl", "device": "mali"},
    {"kind": "opencl", "device": "intel_graphics"},
    "metal",
    "rocm",
    "hexagon",
]


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
    return any(target_kind == t["target_kind"] for t in _get_targets() if t["is_runnable"])


def enabled_targets():
    """Get all enabled targets with associated devices.

    In most cases, parametrize over the specific targets you need with
    ``@pytest.mark.parametrize`` instead of iterating this function.

    In this context, enabled means that TVM was built with support for
    this target, the target name appears in the TVM_TEST_TARGETS
    environment variable, and a suitable device for running this
    target exists.  If TVM_TEST_TARGETS is not set, it defaults to
    variable DEFAULT_TEST_TARGETS in this module.

    If you use this function in a test, you **must** mark the test with
    ``@pytest.mark.gpu`` (otherwise it will never be run on the gpu).

    Returns
    -------
    targets: list
        A list of pairs of all enabled devices and the associated context

    """
    return [
        (t["target"], tvm.device_from_target(t["target"]))
        for t in _get_targets()
        if t["is_runnable"]
    ]


def _parse_target_entry(entry):
    """Parse a target entry from TVM_TEST_TARGETS env var.

    Entries can be plain kind names (e.g. "llvm") or JSON dicts
    (e.g. '{"kind": "opencl", "device": "mali"}').
    """
    entry = entry.strip()
    if entry.startswith("{"):
        import json  # pylint: disable=import-outside-toplevel

        return json.loads(entry)
    return entry


def _tvm_test_targets():
    target_str = os.environ.get("TVM_TEST_TARGETS", "").strip()
    if target_str:
        # De-duplicate while preserving order. dict items can't be hashed
        # directly, so use their str() form as the dedup key.
        targets = []
        seen = set()
        for t in target_str.split(";"):
            t = t.strip()
            if not t:
                continue
            parsed = _parse_target_entry(t)
            key = str(parsed)
            if key in seen:
                continue
            seen.add(key)
            targets.append(parsed)
        return targets

    return DEFAULT_TEST_TARGETS


def _compose(args, decs):
    """Helper to apply multiple markers"""
    if len(args) > 0:
        f = args[0]
        for d in reversed(decs):
            f = d(f)
        return f
    return decs


slow = pytest.mark.skipif(
    SKIP_SLOW_TESTS,
    reason="Skipping slow test since the SKIP_SLOW_TESTS environment variable is 'true'",
)


def skip_if_32bit(reason):
    def decorator(*args):
        if "32bit" in platform.architecture()[0]:
            return _compose(args, [pytest.mark.skip(reason=reason)])

        return _compose(args, [])

    return decorator


def parameter(*values, ids=None, by_dict=None):
    """Convenience function to define pytest parametrized fixtures.

    Declaring a variable using ``tvm.testing.parameter`` will define a
    parametrized pytest fixture that can be used by test
    functions. This is intended for cases that have no setup cost,
    such as strings, integers, tuples, etc.  For cases that have a
    significant setup cost, please use :py:func:`tvm.testing.fixture`
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


def fixture(func=None, *, cache_return_value=False):
    """Convenience function to define pytest fixtures.

    This should be used as a decorator to mark functions that set up
    state before a function.  The return value of that fixture
    function is then accessible by test functions as that accept it as
    a parameter.

    Fixture functions can accept parameters defined with
    :py:func:`tvm.testing.parameter`.

    By default, the setup will be performed once for each unit test
    that uses a fixture, to ensure that unit tests are independent.
    If the setup is expensive to perform, then the
    cache_return_value=True argument can be passed to cache the setup.
    The fixture function will be run only once (or once per parameter,
    if used with tvm.testing.parameter).  The cached setup value is
    retained for the lifetime of the test process, and each test receives
    an independent copy.  If the environment variable TVM_TEST_DISABLE_CACHE
    is set to a non-zero value, it will disable this feature and no caching
    will be performed.

    Example
    -------
    >>> @tvm.testing.fixture
    >>> def cheap_setup():
    >>>     return 5 # Setup code here.
    >>>
    >>> def test_feature_x(target, dev, cheap_setup)
    >>>     assert(cheap_setup == 5) # Run test here

    Or

    >>> size = tvm.testing.parameter(1, 10, 100)
    >>>
    >>> @tvm.testing.fixture
    >>> def cheap_setup(size):
    >>>     return 5*size # Setup code here, based on size.
    >>>
    >>> def test_feature_x(cheap_setup):
    >>>     assert(cheap_setup in [5, 50, 500])

    Or

    >>> @tvm.testing.fixture(cache_return_value=True)
    >>> def expensive_setup():
    >>>     time.sleep(10) # Setup code here
    >>>     return 5
    >>>
    >>> def test_feature_x(target, dev, expensive_setup):
    >>>     assert(expensive_setup == 5)

    """

    force_disable_cache = bool(int(os.environ.get("TVM_TEST_DISABLE_CACHE", "0")))
    cache_return_value = cache_return_value and not force_disable_cache

    def wraps(func):
        if cache_return_value:
            func = _fixture_cache(func)
        func = pytest.fixture(func, scope="function")
        return func

    if func is None:
        return wraps

    return wraps(func)


class _DeepCopyAllowedClasses(dict):
    def __init__(self, allowed_class_list):
        self.allowed_class_list = allowed_class_list
        super().__init__()

    def get(self, key, *args, **kwargs):
        """Overrides behavior of copy.deepcopy to avoid implicit copy.

        By default, copy.deepcopy uses a dict of id->object to track
        all objects that it has seen, which is passed as the second
        argument to all recursive calls.  This class is intended to be
        passed in instead, and inspects the type of all objects being
        copied.

        Where copy.deepcopy does a best-effort attempt at copying an
        object, for unit tests we would rather have all objects either
        be copied correctly, or to throw an error.  Classes that
        define an explicit method to perform a copy are allowed, as
        are any explicitly listed classes.  Classes that would fall
        back to using object.__reduce__, and are not explicitly listed
        as safe, will throw an exception.

        """
        obj = ctypes.cast(key, ctypes.py_object).value
        cls = type(obj)
        if (
            cls in copy._deepcopy_dispatch
            or issubclass(cls, type)
            or getattr(obj, "__deepcopy__", None)
            or copyreg.dispatch_table.get(cls)
            or cls.__reduce__ is not object.__reduce__
            or cls.__reduce_ex__ is not object.__reduce_ex__
            or cls in self.allowed_class_list
        ):
            return super().get(key, *args, **kwargs)

        rfc_url = (
            "https://github.com/apache/tvm-rfcs/blob/main/rfcs/0007-parametrized-unit-tests.md"
        )
        raise TypeError(
            f"Cannot copy fixture of type {cls.__name__}.  TVM fixture caching "
            "is limited to objects that explicitly provide the ability "
            "to be copied (e.g. through __deepcopy__, __getstate__, or __setstate__),"
            "and forbids the use of the default `object.__reduce__` and "
            "`object.__reduce_ex__`.  For third-party classes that are "
            "safe to use with copy.deepcopy, please add the class to "
            "the arguments of _DeepCopyAllowedClasses in tvm.testing._fixture_cache.\n"
            "\n"
            f"For discussion on this restriction, please see {rfc_url}."
        )


def _fixture_cache(func):
    cache = {}

    # Using functools.lru_cache would require the function arguments
    # to be hashable, which wouldn't allow caching fixtures that
    # depend on numpy arrays.  For example, a fixture that takes a
    # numpy array as input, then calculates uses a slow method to
    # compute a known correct output for that input.  Therefore,
    # including a fallback for serializable types.
    def get_cache_key(*args, **kwargs):
        try:
            hash((args, kwargs))
            return (args, kwargs)
        except TypeError:
            pass

        try:
            return pickle.dumps((args, kwargs))
        except TypeError as e:
            raise TypeError(
                "TVM caching of fixtures requires arguments to the fixture "
                "to be either hashable or serializable"
            ) from e

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = get_cache_key(*args, **kwargs)

        try:
            cached_value = cache[cache_key]
        except KeyError:
            cached_value = cache[cache_key] = func(*args, **kwargs)

        return copy.deepcopy(
            cached_value,
            # allowed_class_list should be a list of classes that
            # are safe to copy using copy.deepcopy, but do not
            # implement __deepcopy__, __reduce__, or
            # __reduce_ex__.
            _DeepCopyAllowedClasses(allowed_class_list=[]),
        )

    return wrapper


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


def is_ampere_or_newer():
    """Check if the target environment has an NVIDIA Ampere GPU or newer."""
    arch = nvcc.get_target_compute_version()
    major, minor = nvcc.parse_compute_version(arch)
    return major >= 8 and minor != 9


def install_request_hook(hook_script: Path) -> None:
    """Add a wrapper around urllib.request for CI tests."""
    if not IS_IN_CI:
        return

    hook_script = Path(hook_script).resolve()
    if not hook_script.is_file():
        raise RuntimeError(f"Request hook {hook_script} does not exist")

    # Load the exact hook file without exposing the test root as an import path.
    # Cache its initializer because Sphinx invokes this once per gallery example.
    try:
        init = _REQUEST_HOOK_INITIALIZERS[hook_script]
    except KeyError:
        init = _REQUEST_HOOK_INITIALIZERS[hook_script] = runpy.run_path(str(hook_script))["init"]
    init()


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")


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
