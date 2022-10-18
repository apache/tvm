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

# pylint: disable=invalid-name,unnecessary-comprehension
"""TVM testing utilities

Organization
************

This file contains functions expected to be called directly by a user
while writing unit tests.  Integrations with the pytest framework
are in plugin.py.

Testing Markers
***************

We use pytest markers to specify the requirements of test functions. Currently
there is a single distinction that matters for our testing environment: does
the test require a gpu. For tests that require just a gpu or just a cpu, we
have the decorator :py:func:`requires_gpu` that enables the test when a gpu is
available. To avoid running tests that don't require a gpu on gpu nodes, this
decorator also sets the pytest marker `gpu` so we can use select the gpu subset
of tests (using `pytest -m gpu`).

Unfortunately, many tests are written like this:

.. python::

    def test_something():
        for target in all_targets():
            do_something()

The test uses both gpu and cpu targets, so the test needs to be run on both cpu
and gpu nodes. But we still want to only run the cpu targets on the cpu testing
node. The solution is to mark these tests with the gpu marker so they will be
run on the gpu nodes. But we also modify all_targets (renamed to
enabled_targets) so that it only returns gpu targets on gpu nodes and cpu
targets on cpu nodes (using an environment variable).

Instead of using the all_targets function, future tests that would like to
test against a variety of targets should use the
:py:func:`tvm.testing.parametrize_targets` functionality. This allows us
greater control over which targets are run on which testing nodes.

If in the future we want to add a new type of testing node (for example
fpgas), we need to add a new marker in `tests/python/pytest.ini` and a new
function in this module. Then targets using this node should be added to the
`TVM_TEST_TARGETS` environment variable in the CI.

"""
import inspect
import copy
import copyreg
import ctypes
import functools
import hashlib
import itertools
import logging
import os
import pickle
import platform
import sys
import textwrap
import time
import shutil
import subprocess

from pathlib import Path
from typing import Optional, Callable, Union, List, Tuple

import pytest
import numpy as np

import tvm
import tvm.arith
import tvm.tir
import tvm.te
import tvm._ffi

from tvm.contrib import nvcc, cudnn
import tvm.contrib.hexagon._ci_env_check as hexagon
from tvm.driver.tvmc.frontends import load_model
from tvm.error import TVMError


SKIP_SLOW_TESTS = os.getenv("SKIP_SLOW_TESTS", "").lower() in {"true", "1", "yes"}
IS_IN_CI = os.getenv("CI", "") == "true"

skip_if_wheel_test = pytest.mark.skipif(
    os.getenv("WHEEL_TEST", "").lower() in {"true", "1", "yes"},
    reason="Test not supported in wheel.",
)


def assert_allclose(actual, desired, rtol=1e-7, atol=1e-7):
    """Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `actual` and `desired` are not interchangeable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """
    actual = np.asanyarray(actual)
    desired = np.asanyarray(desired)
    np.testing.assert_allclose(actual.shape, desired.shape)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=True)


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
                "Gradient wrt '{}' has unexpected shape {}, expected {} ".format(
                    x_name, grad.shape, input_values[x_name].shape
                )
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
                "NaN or infinity detected during numerical gradient checking wrt '{}'\n"
                "analytical grad = {}\n numerical grad = {}\n".format(x_name, grad, ngrad)
            )

        # we multiply atol by this number to make it more universal for different sizes
        sqrt_n = np.sqrt(float(np.prod(grad.shape)))

        if dist > atol * sqrt_n + rtol * grad_norm:
            raise AssertionError(
                "Analytical and numerical grads wrt '{}' differ too much\n"
                "analytical grad = {}\n numerical grad = {}\n"
                "{}% of elements differ, first 10 of wrong positions: {}\n"
                "distance > atol*sqrt(n) + rtol*grad_norm\n"
                "distance {} > {}*{} + {}*{}".format(
                    x_name,
                    grad,
                    ngrad,
                    wrong_percentage,
                    wrong_positions[:10],
                    dist,
                    atol,
                    sqrt_n,
                    rtol,
                    grad_norm,
                )
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
    lhs : tvm.tir.PrimExpr
        The left operand.

    rhs : tvm.tir.PrimExpr
        The left operand.
    """
    ana = tvm.arith.Analyzer()
    if not ana.can_prove_equal(lhs, rhs):
        raise ValueError("{} and {} are not equal".format(lhs, rhs))


def check_bool_expr_is_true(bool_expr, vranges, cond=None):
    """Check that bool_expr holds given the condition cond
    for every value of free variables from vranges.

    for example, 2x > 4y solves to x > 2y given x in (0, 10) and y in (0, 10)
    here bool_expr is x > 2y, vranges is {x: (0, 10), y: (0, 10)}, cond is 2x > 4y
    We creates iterations to check,
    for x in range(10):
      for y in range(10):
        assert !(2x > 4y) || (x > 2y)

    Parameters
    ----------
    bool_expr : tvm.ir.PrimExpr
        Boolean expression to check
    vranges: Dict[tvm.tir.expr.Var, tvm.ir.Range]
        Free variables and their ranges
    cond: tvm.ir.PrimExpr
        extra conditions needs to be satisfied.
    """
    if cond is not None:
        bool_expr = tvm.te.any(tvm.tir.Not(cond), bool_expr)

    def _run_expr(expr, vranges):
        """Evaluate expr for every value of free variables
        given by vranges and return the tensor of results.
        """

        def _compute_body(*us):
            vmap = {v: u + r.min for (v, r), u in zip(vranges.items(), us)}
            return tvm.tir.stmt_functor.substitute(expr, vmap)

        A = tvm.te.compute([r.extent.value for v, r in vranges.items()], _compute_body)
        args = [tvm.nd.empty(A.shape, A.dtype)]
        sch = tvm.te.create_schedule(A.op)
        mod = tvm.build(sch, [A])
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
            "Expression {}\nis not true on {}\n"
            "Counterexample: {}".format(ana.simplify(bool_expr), vranges, counterex)
        )


def check_int_constraints_trans_consistency(constraints_trans, vranges=None):
    """Check IntConstraintsTransform is a bijective transformation.

    Parameters
    ----------
    constraints_trans : arith.IntConstraintsTransform
        Integer constraints transformation
    vranges: Dict[tvm.tir.Var, tvm.ir.Range]
        Free variables and their ranges
    """
    if vranges is None:
        vranges = {}

    def _check_forward(constraints1, constraints2, varmap, backvarmap):
        ana = tvm.arith.Analyzer()
        all_vranges = vranges.copy()
        all_vranges.update({v: r for v, r in constraints1.ranges.items()})

        # Check that the transformation is injective
        cond_on_vars = tvm.tir.const(1, "bool")
        for v in constraints1.variables:
            if v in varmap:
                # variable mapping is consistent
                v_back = ana.simplify(tvm.tir.stmt_functor.substitute(varmap[v], backvarmap))
                cond_on_vars = tvm.te.all(cond_on_vars, v == v_back)
        # Also we have to check that the new relations are true when old relations are true
        cond_subst = tvm.tir.stmt_functor.substitute(
            tvm.te.all(tvm.tir.const(1, "bool"), *constraints2.relations), backvarmap
        )
        # We have to include relations from vranges too
        for v in constraints2.variables:
            if v in constraints2.ranges:
                r = constraints2.ranges[v]
                range_cond = tvm.te.all(v >= r.min, v < r.min + r.extent)
                range_cond = tvm.tir.stmt_functor.substitute(range_cond, backvarmap)
                cond_subst = tvm.te.all(cond_subst, range_cond)
        cond_subst = ana.simplify(cond_subst)
        check_bool_expr_is_true(
            tvm.te.all(cond_subst, cond_on_vars),
            all_vranges,
            cond=tvm.te.all(tvm.tir.const(1, "bool"), *constraints1.relations),
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
        target_kind = target.split()[0]

        if target_kind == "cuda" and "cudnn" in tvm.target.Target(target).attrs.get("libs", []):
            is_enabled = tvm.support.libinfo()["USE_CUDNN"].lower() in ["on", "true", "1"]
            is_runnable = is_enabled and cudnn.exists()
        elif target_kind == "hexagon":
            is_enabled = tvm.support.libinfo()["USE_HEXAGON"].lower() in ["on", "true", "1"]
            # If Hexagon has compile-time support, we can always fall back
            is_runnable = is_enabled and "ANDROID_SERIAL_NUMBER" in os.environ
        else:
            is_enabled = tvm.runtime.enabled(target_kind)
            is_runnable = is_enabled and tvm.device(target_kind).exist

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

        raise TVMError(
            "None of the following targets are supported by this build of TVM: %s."
            " Try setting TVM_TEST_TARGETS to a supported target."
            " Cannot default to llvm, as it is not enabled." % target_names
        )

    return targets


DEFAULT_TEST_TARGETS = [
    "llvm",
    "cuda",
    "nvptx",
    "vulkan -from_device=0",
    "opencl",
    "opencl -device=mali,aocl_sw_emu",
    "opencl -device=intel_graphics",
    "metal",
    "rocm",
    "hexagon",
]


def device_enabled(target):
    """Check if a target should be used when testing.

    It is recommended that you use :py:func:`tvm.testing.parametrize_targets`
    instead of manually checking if a target is enabled.

    This allows the user to control which devices they are testing against. In
    tests, this should be used to check if a device should be used when said
    device is an optional part of the test.

    Parameters
    ----------
    target : str
        Target string to check against

    Returns
    -------
    bool
        Whether or not the device associated with this target is enabled.

    Example
    -------
    >>> @tvm.testing.uses_gpu
    >>> def test_mytest():
    >>>     for target in ["cuda", "llvm"]:
    >>>         if device_enabled(target):
    >>>             test_body...

    Here, `test_body` will only be reached by with `target="cuda"` on gpu test
    nodes and `target="llvm"` on cpu test nodes.
    """
    assert isinstance(target, str), "device_enabled requires a target as a string"
    # only check if device name is found, sometime there are extra flags
    target_kind = target.split(" ")[0]
    return any(target_kind == t["target_kind"] for t in _get_targets() if t["is_runnable"])


def enabled_targets():
    """Get all enabled targets with associated devices.

    In most cases, you should use :py:func:`tvm.testing.parametrize_targets` instead of
    this function.

    In this context, enabled means that TVM was built with support for
    this target, the target name appears in the TVM_TEST_TARGETS
    environment variable, and a suitable device for running this
    target exists.  If TVM_TEST_TARGETS is not set, it defaults to
    variable DEFAULT_TEST_TARGETS in this module.

    If you use this function in a test, you **must** decorate the test with
    :py:func:`tvm.testing.uses_gpu` (otherwise it will never be run on the gpu).

    Returns
    -------
    targets: list
        A list of pairs of all enabled devices and the associated context

    """
    return [(t["target"], tvm.device(t["target"])) for t in _get_targets() if t["is_runnable"]]


class Feature:

    """A feature that may be required to run a test.

    Parameters
    ----------
    name: str

        The short name of the feature.  Should match the name in the
        requires_* decorator.  This is applied as a mark to all tests
        using this feature, and can be used in pytests ``-m``
        argument.

    long_name: Optional[str]

        The long name of the feature, to be used in error messages.

        If None, defaults to the short name.

    cmake_flag: Optional[str]

        The flag that must be enabled in the config.cmake in order to
        use this feature.

        If None, no flag is required to use this feature.

    target_kind_enabled: Optional[str]

        The target kind that must be enabled to run tests using this
        feature.  If present, the target_kind must appear in the
        TVM_TEST_TARGETS environment variable, or in
        tvm.testing.DEFAULT_TEST_TARGETS if TVM_TEST_TARGETS is
        undefined.

        If None, this feature does not require a specific target to be
        enabled.

    compile_time_check: Optional[Callable[[], Union[bool,str]]]

        A check that returns True if the feature can be used at
        compile-time.  (e.g. Validating the version number of the nvcc
        compiler.)  If the feature does not have support to perform
        compile-time tests, the check should returns False to display
        a generic error message, or a string to display a more
        specific error message.

        If None, no additional check is performed.

    target_kind_hardware: Optional[str]

        The target kind that must have available hardware in order to
        run tests using this feature.  This is checked using
        tvm.device(target_kind_hardware).exist.  If a feature requires
        a different check, this should be implemented using
        run_time_check.

        If None, this feature does not require a specific
        tvm.device to exist.

    run_time_check: Optional[Callable[[], Union[bool,str]]]

        A check that returns True if the feature can be used at
        run-time.  (e.g. Validating the compute version supported by a
        GPU.)  If the feature does not have support to perform
        run-time tests, the check should returns False to display a
        generic error message, or a string to display a more specific
        error message.

        If None, no additional check is performed.

    parent_features: Optional[Union[str,List[str]]]

        The short name of a feature or features that are required in
        order to use this feature.  (e.g. Using cuDNN requires using
        CUDA) This feature should inherit all checks of the parent
        feature, with the exception of the `target_kind_enabled`
        checks.

        If None, this feature does not require any other parent
        features.

    """

    _all_features = {}

    def __init__(
        self,
        name: str,
        long_name: Optional[str] = None,
        cmake_flag: Optional[str] = None,
        target_kind_enabled: Optional[str] = None,
        compile_time_check: Optional[Callable[[], Union[bool, str]]] = None,
        target_kind_hardware: Optional[str] = None,
        run_time_check: Optional[Callable[[], Union[bool, str]]] = None,
        parent_features: Optional[Union[str, List[str]]] = None,
    ):
        self.name = name
        self.long_name = long_name or name
        self.cmake_flag = cmake_flag
        self.target_kind_enabled = target_kind_enabled
        self.compile_time_check = compile_time_check
        self.target_kind_hardware = target_kind_hardware
        self.run_time_check = run_time_check

        if parent_features is None:
            self.parent_features = []
        elif isinstance(parent_features, str):
            self.parent_features = [parent_features]
        else:
            self.parent_features = parent_features

        self._all_features[self.name] = self

    def _register_marker(self, config):
        config.addinivalue_line("markers", f"{self.name}: Mark a test as using {self.long_name}")

    def _uses_marks(self):
        for parent in self.parent_features:
            yield from self._all_features[parent]._uses_marks()

        yield getattr(pytest.mark, self.name)

    def _compile_only_marks(self):
        for parent in self.parent_features:
            yield from self._all_features[parent]._compile_only_marks()

        if self.compile_time_check is not None:
            res = self.compile_time_check()
            if isinstance(res, str):
                yield pytest.mark.skipif(True, reason=res)
            else:
                yield pytest.mark.skipif(
                    not res, reason=f"Compile-time support for {self.long_name} not present"
                )

        if self.target_kind_enabled is not None:
            target_kind = self.target_kind_enabled.split()[0]
            yield pytest.mark.skipif(
                all(enabled.split()[0] != target_kind for enabled in _tvm_test_targets()),
                reason=(
                    f"{self.target_kind_enabled} tests disabled "
                    f"by TVM_TEST_TARGETS environment variable"
                ),
            )

        if self.cmake_flag is not None:
            yield pytest.mark.skipif(
                not _cmake_flag_enabled(self.cmake_flag),
                reason=(
                    f"{self.long_name} support not enabled.  "
                    f"Set {self.cmake_flag} in config.cmake to enable."
                ),
            )

    def _run_only_marks(self):
        for parent in self.parent_features:
            yield from self._all_features[parent]._run_only_marks()

        if self.run_time_check is not None:
            res = self.run_time_check()
            if isinstance(res, str):
                yield pytest.mark.skipif(True, reason=res)
            else:
                yield pytest.mark.skipif(
                    not res, reason=f"Run-time support for {self.long_name} not present"
                )

        if self.target_kind_hardware is not None:
            yield pytest.mark.skipif(
                not tvm.device(self.target_kind_hardware).exist,
                reason=f"No device exists for target {self.target_kind_hardware}",
            )

    def marks(self, support_required="compile-and-run"):
        """Return a list of marks to be used

        Parameters
        ----------

        support_required: str

            Allowed values: "compile-and-run" (default),
            "compile-only", or "optional".

            See Feature.__call__ for details.
        """
        if support_required not in ["compile-and-run", "compile-only", "optional"]:
            raise ValueError(f"Unknown feature support type: {support_required}")

        if support_required == "compile-and-run":
            marks = itertools.chain(
                self._run_only_marks(), self._compile_only_marks(), self._uses_marks()
            )
        elif support_required == "compile-only":
            marks = itertools.chain(self._compile_only_marks(), self._uses_marks())
        elif support_required == "optional":
            marks = self._uses_marks()
        else:
            raise ValueError(f"Unknown feature support type: {support_required}")

        return list(marks)

    def __call__(self, func=None, *, support_required="compile-and-run"):
        """Mark a pytest function as requiring this feature

        Can be used either as a bare decorator, or as a decorator with
        arguments.

        Parameters
        ----------

        func: Callable

            The pytest test function to be marked

        support_required: str

            Allowed values: "compile-and-run" (default),
            "compile-only", or "optional".

            If "compile-and-run", the test case is marked as using the
            feature, and is skipped if the environment lacks either
            compile-time or run-time support for the feature.

            If "compile-only", the test case is marked as using the
            feature, and is skipped if the environment lacks
            compile-time support.

            If "optional", the test case is marked as using the
            feature, but isn't skipped.  This is kept for backwards
            compatibility for tests that use `enabled_targets()`, and
            should be avoided in new test code.  Instead, prefer
            parametrizing over the target using the `target` fixture.

        Examples
        --------

        .. code-block:: python

          @feature
          def test_compile_and_run():
              ...

          @feature(compile_only=True)
          def test_compile_only():
              ...

        """

        if support_required not in ["compile-and-run", "compile-only", "optional"]:
            raise ValueError(f"Unknown feature support type: {support_required}")

        def wrapper(func):
            for mark in self.marks(support_required=support_required):
                func = mark(func)
            return func

        if func is None:
            return wrapper

        return wrapper(func)

    @classmethod
    def require(cls, name, support_required="compile-and-run"):
        """Returns a decorator that marks a test as requiring a feature

        Parameters
        ----------

        name: str

            The name of the feature that is used by the test

        support_required: str

            Allowed values: "compile-and-run" (default),
            "compile-only", or "optional".

            See Feature.__call__ for details.

        Examples
        --------

        .. code-block:: python

          @Feature.require("cuda")
          def test_compile_and_run():
              ...

          @Feature.require("cuda", compile_only=True)
          def test_compile_only():
              ...
        """
        return cls._all_features[name](support_required=support_required)


def _any_gpu_exists():
    return (
        tvm.cuda().exist
        or tvm.rocm().exist
        or tvm.opencl().exist
        or tvm.metal().exist
        or tvm.vulkan().exist
    )


# Mark a test as requiring llvm to run
requires_llvm = Feature(
    "llvm", "LLVM", cmake_flag="USE_LLVM", target_kind_enabled="llvm", target_kind_hardware="llvm"
)

# Mark a test as requiring a GPU to run.
requires_gpu = Feature("gpu", run_time_check=_any_gpu_exists)

# Mark to differentiate tests that use the GPU in some capacity.
#
# These tests will be run on CPU-only test nodes and on test nodes with GPUs.
# To mark a test that must have a GPU present to run, use
# :py:func:`tvm.testing.requires_gpu`.
uses_gpu = requires_gpu(support_required="optional")

# Mark a test as requiring the x86 Architecture to run.
requires_x86 = Feature(
    "x86", "x86 Architecture", run_time_check=lambda: platform.machine() == "x86_64"
)

# Mark a test as requiring the CUDA runtime.
requires_cuda = Feature(
    "cuda",
    "CUDA",
    cmake_flag="USE_CUDA",
    target_kind_enabled="cuda",
    target_kind_hardware="cuda",
    parent_features="gpu",
)

# Mark a test as requiring a tensorcore to run
requires_tensorcore = Feature(
    "tensorcore",
    "NVIDIA Tensor Core",
    run_time_check=lambda: tvm.cuda().exist and nvcc.have_tensorcore(tvm.cuda().compute_version),
    parent_features="cuda",
)

# Mark a test as requiring the cuDNN library.
requires_cudnn = Feature("cudnn", "cuDNN", cmake_flag="USE_CUDNN", parent_features="cuda")

# Mark a test as requiring the cuBLAS library.
requires_cublas = Feature("cublas", "cuBLAS", cmake_flag="USE_CUBLAS", parent_features="cuda")

# Mark a test as requiring the NVPTX compilation on the CUDA runtime
requires_nvptx = Feature(
    "nvptx",
    "NVPTX",
    target_kind_enabled="nvptx",
    target_kind_hardware="nvptx",
    parent_features=["llvm", "cuda"],
)

# Mark a test as requiring the CUDA Graph Feature
requires_cudagraph = Feature(
    "cudagraph",
    "CUDA Graph",
    target_kind_enabled="cuda",
    compile_time_check=nvcc.have_cudagraph,
    parent_features="cuda",
)

# Mark a test as requiring the OpenCL runtime
requires_opencl = Feature(
    "opencl",
    "OpenCL",
    cmake_flag="USE_OPENCL",
    target_kind_enabled="opencl",
    target_kind_hardware="opencl" if "RPC_TARGET" not in os.environ else None,
    parent_features="gpu" if "RPC_TARGET" not in os.environ else None,
)

# Mark a test as requiring the rocm runtime
requires_rocm = Feature(
    "rocm",
    "ROCm",
    cmake_flag="USE_ROCM",
    target_kind_enabled="rocm",
    target_kind_hardware="rocm",
    parent_features="gpu",
)

# Mark a test as requiring the metal runtime
requires_metal = Feature(
    "metal",
    "Metal",
    cmake_flag="USE_METAL",
    target_kind_enabled="metal",
    target_kind_hardware="metal",
    parent_features="gpu",
)

# Mark a test as requiring the vulkan runtime
requires_vulkan = Feature(
    "vulkan",
    "Vulkan",
    cmake_flag="USE_VULKAN",
    target_kind_enabled="vulkan",
    target_kind_hardware="vulkan",
    parent_features="gpu",
)

# Mark a test as requiring OpenCLML support in build.
requires_openclml = Feature(
    "OpenCLML",
    "CLML",
    cmake_flag="USE_CLML",
    target_kind_enabled="opencl",
)


# Mark a test as requiring microTVM to run
requires_micro = Feature("micro", "MicroTVM", cmake_flag="USE_MICRO")

# Mark a test as requiring CUTLASS to run
requires_cutlass = Feature("cutlass", "CUTLASS", cmake_flag="USE_CUTLASS")

# Mark a test as requiring rpc to run
requires_rpc = Feature("rpc", "RPC", cmake_flag="USE_RPC")

# Mark a test as requiring Arm(R) Ethos(TM)-N to run
requires_ethosn = Feature("ethosn", "Arm(R) Ethos(TM)-N", cmake_flag="USE_ETHOSN")

# Mark a test as requiring libtorch to run
requires_libtorch = Feature("libtorch", "LibTorch", cmake_flag="USE_LIBTORCH")

# Mark a test as requiring Hexagon to run
requires_hexagon = Feature(
    "hexagon",
    "Hexagon",
    cmake_flag="USE_HEXAGON",
    target_kind_enabled="hexagon",
    compile_time_check=hexagon._compile_time_check,
    run_time_check=hexagon._run_time_check,
    parent_features="llvm",
)

# Mark a test as requiring the CMSIS NN library
requires_cmsisnn = Feature("cmsisnn", "CMSIS NN", cmake_flag="USE_CMSISNN")


def _corstone300_compile_time_check():
    if shutil.which("arm-none-eabi-gcc") is None:
        return "ARM embedded toolchain unavailable"
    return True


# Mark a test as requiring the corstone300 FVP
requires_corstone300 = Feature(
    "corstone300",
    "Corstone-300",
    compile_time_check=_corstone300_compile_time_check,
    parent_features="cmsisnn",
)

# Mark a test as requiring Vitis AI to run
requires_vitis_ai = Feature("vitis_ai", "Vitis AI", cmake_flag="USE_VITIS_AI")


def _arm_dot_supported():
    arch = platform.machine()

    if arch not in ["arm64", "aarch64"]:
        return False

    if sys.platform.startswith("darwin"):
        cpu_info = subprocess.check_output("sysctl -a", shell=True).strip().decode()
        for line in cpu_info.split("\n"):
            if line.startswith("hw.optional.arm.FEAT_DotProd"):
                return bool(int(line.split(":", 1)[1]))
    elif sys.platform.startswith("linux"):
        return True

    return False


def _is_intel():
    # Only linux is supported for now.
    if sys.platform.startswith("linux"):
        with open("/proc/cpuinfo", "r") as content:
            return "Intel" in content.read()

    return False


def _has_vnni():
    arch = platform.machine()
    # Only linux is supported for now.
    if arch == "x86_64" and sys.platform.startswith("linux"):
        with open("/proc/cpuinfo", "r") as content:
            return "avx512_vnni" in content.read()

    return False


requires_arm_dot = Feature("arm_dot", "ARM dot product", run_time_check=_arm_dot_supported)


requires_cascadelake = Feature(
    "cascadelake", "x86 CascadeLake", run_time_check=lambda: _has_vnni() and _is_intel()
)


def _cmake_flag_enabled(flag):
    flag = tvm.support.libinfo()[flag]

    # Because many of the flags can be library flags, we check if the
    # flag is not disabled, rather than checking if it is enabled.
    return flag.lower() not in ["off", "false", "0"]


def _tvm_test_targets():
    target_str = os.environ.get("TVM_TEST_TARGETS", "").strip()
    if target_str:
        # Use dict instead of set for de-duplication so that the
        # targets stay in the order specified.
        return list({t.strip(): None for t in target_str.split(";") if t.strip()})

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


def requires_nvcc_version(major_version, minor_version=0, release_version=0):
    """Mark a test as requiring at least a specific version of nvcc.

    Unit test marked with this decorator will run only if the
    installed version of NVCC is at least `(major_version,
    minor_version, release_version)`.

    This also marks the test as requiring a cuda support.

    Parameters
    ----------
    major_version: int

        The major version of the (major,minor,release) version tuple.

    minor_version: int

        The minor version of the (major,minor,release) version tuple.

    release_version: int

        The release version of the (major,minor,release) version tuple.

    """

    try:
        nvcc_version = nvcc.get_cuda_version()
    except RuntimeError:
        nvcc_version = (0, 0, 0)

    min_version = (major_version, minor_version, release_version)
    version_str = ".".join(str(v) for v in min_version)
    requires = [
        pytest.mark.skipif(nvcc_version < min_version, reason=f"Requires NVCC >= {version_str}"),
        *requires_cuda.marks(),
    ]

    def inner(func):
        return _compose([func], requires)

    return inner


def requires_cuda_compute_version(major_version, minor_version=0):
    """Mark a test as requiring at least a compute architecture

    Unit test marked with this decorator will run only if the CUDA
    compute architecture of the GPU is at least `(major_version,
    minor_version)`.

    This also marks the test as requiring a cuda support.

    Parameters
    ----------
    major_version: int

        The major version of the (major,minor) version tuple.

    minor_version: int

        The minor version of the (major,minor) version tuple.
    """
    min_version = (major_version, minor_version)
    try:
        arch = tvm.contrib.nvcc.get_target_compute_version()
        compute_version = tvm.contrib.nvcc.parse_compute_version(arch)
    except ValueError:
        # No GPU present.  This test will be skipped from the
        # requires_cuda() marks as well.
        compute_version = (0, 0)

    min_version_str = ".".join(str(v) for v in min_version)
    compute_version_str = ".".join(str(v) for v in compute_version)
    requires = [
        pytest.mark.skipif(
            compute_version < min_version,
            reason=f"Requires CUDA compute >= {min_version_str}, but have {compute_version_str}",
        ),
        *requires_cuda.marks(),
    ]

    def inner(func):
        return _compose([func], requires)

    return inner


def skip_if_32bit(reason):
    def decorator(*args):
        if "32bit" in platform.architecture()[0]:
            return _compose(args, [pytest.mark.skip(reason=reason)])

        return _compose(args, [])

    return decorator


def requires_package(*packages):
    """Mark a test as requiring python packages to run.

    If the packages listed are not available, tests marked with
    `requires_package` will appear in the pytest results as being skipped.
    This is equivalent to using ``foo = pytest.importorskip('foo')`` inside
    the test body.

    Parameters
    ----------
    packages : List[str]

        The python packages that should be available for the test to
        run.

    Returns
    -------
    mark: pytest mark

        The pytest mark to be applied to unit tests that require this

    """

    def has_package(package):
        try:
            __import__(package)
            return True
        except ImportError:
            return False

    marks = [
        pytest.mark.skipif(not has_package(package), reason=f"Cannot import '{package}'")
        for package in packages
    ]

    def wrapper(func):
        for mark in marks:
            func = mark(func)
        return func

    return wrapper


def parametrize_targets(*args):

    """Parametrize a test over a specific set of targets.

    Use this decorator when you want your test to be run over a
    specific set of targets and devices.  It is intended for use where
    a test is applicable only to a specific target, and is
    inapplicable to any others (e.g. verifying target-specific
    assembly code matches known assembly code).  In most
    circumstances, :py:func:`tvm.testing.exclude_targets` or
    :py:func:`tvm.testing.known_failing_targets` should be used
    instead.

    If used as a decorator without arguments, the test will be
    parametrized over all targets in
    :py:func:`tvm.testing.enabled_targets`.  This behavior is
    automatically enabled for any target that accepts arguments of
    ``target`` or ``dev``, so the explicit use of the bare decorator
    is no longer needed, and is maintained for backwards
    compatibility.

    Parameters
    ----------
    f : function
        Function to parametrize. Must be of the form `def test_xxxxxxxxx(target, dev)`:,
        where `xxxxxxxxx` is any name.
    targets : list[str], optional
        Set of targets to run against. If not supplied,
        :py:func:`tvm.testing.enabled_targets` will be used.

    Example
    -------
    >>> @tvm.testing.parametrize_targets("llvm", "cuda")
    >>> def test_mytest(target, dev):
    >>>     ...  # do something
    """

    # Backwards compatibility, when used as a decorator with no
    # arguments implicitly parametrizes over "target".  The
    # parametrization is now handled by _auto_parametrize_target, so
    # this use case can just return the decorated function.
    if len(args) == 1 and callable(args[0]):
        return args[0]

    return pytest.mark.parametrize("target", list(args), scope="session")


def exclude_targets(*args):
    """Exclude a test from running on a particular target.

    Use this decorator when you want your test to be run over a
    variety of targets and devices (including cpu and gpu devices),
    but want to exclude some particular target or targets.  For
    example, a test may wish to be run against all targets in
    tvm.testing.enabled_targets(), except for a particular target that
    does not support the capabilities.

    Applies pytest.mark.skipif to the targets given.

    Parameters
    ----------
    f : function
        Function to parametrize. Must be of the form `def test_xxxxxxxxx(target, dev)`:,
        where `xxxxxxxxx` is any name.
    targets : list[str]
        Set of targets to exclude.

    Example
    -------
    >>> @tvm.testing.exclude_targets("cuda")
    >>> def test_mytest(target, dev):
    >>>     ...  # do something

    Or

    >>> @tvm.testing.exclude_targets("llvm", "cuda")
    >>> def test_mytest(target, dev):
    >>>     ...  # do something

    """

    def wraps(func):
        func.tvm_excluded_targets = args
        return func

    return wraps


def known_failing_targets(*args):
    """Skip a test that is known to fail on a particular target.

    Use this decorator when you want your test to be run over a
    variety of targets and devices (including cpu and gpu devices),
    but know that it fails for some targets.  For example, a newly
    implemented runtime may not support all features being tested, and
    should be excluded.

    Applies pytest.mark.xfail to the targets given.

    Parameters
    ----------
    f : function
        Function to parametrize. Must be of the form `def test_xxxxxxxxx(target, dev)`:,
        where `xxxxxxxxx` is any name.
    targets : list[str]
        Set of targets to skip.

    Example
    -------
    >>> @tvm.testing.known_failing_targets("cuda")
    >>> def test_mytest(target, dev):
    >>>     ...  # do something

    Or

    >>> @tvm.testing.known_failing_targets("llvm", "cuda")
    >>> def test_mytest(target, dev):
    >>>     ...  # do something

    """

    def wraps(func):
        func.tvm_known_failing_targets = args
        return func

    return wraps


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
    @pytest.fixture(params=values, ids=ids)
    def as_fixture(*_cls, request):
        return request.param

    return as_fixture


_parametrize_group = 0


def parameters(*value_sets, ids=None):
    """Convenience function to define pytest parametrized fixtures.

    Declaring a variable using tvm.testing.parameters will define a
    parametrized pytest fixture that can be used by test
    functions. Like :py:func:`tvm.testing.parameter`, this is intended
    for cases that have no setup cost, such as strings, integers,
    tuples, etc.  For cases that have a significant setup cost, please
    use :py:func:`tvm.testing.fixture` instead.

    Unlike :py:func:`tvm.testing.parameter`, if a test function
    accepts multiple parameters defined using a single call to
    ``tvm.testing.parameters``, then the test will only be run once
    for each set of parameters, not for all combinations of
    parameters.

    These parameter definitions apply to all tests in a module.  If a
    specific test should have different values for some parameters,
    that test should be marked with ``@pytest.mark.parametrize``.

    Parameters
    ----------
    values : List[tuple]

       A list of parameter value sets.  Each set of values represents
       a single combination of values to be tested.  A unit test that
       accepts parameters defined will be run once for every set of
       parameters in the list.

    ids : List[str], optional

       A list of names for the parameter sets.  If None, pytest will
       generate a name from each parameter set.  These generated names may
       not be readable/useful for composite types such as tuples.

    Returns
    -------
    List[function]
       Function outputs from pytest.fixture.  These should be unpacked
       into individual named parameters.

    Example
    -------
    >>> size, dtype = tvm.testing.parameters( (16,'float32'), (512,'float16') )
    >>> def test_feature_x(size, dtype):
    >>>     # Test code here
    >>>     assert( (size,dtype) in [(16,'float32'), (512,'float16')])

    """
    global _parametrize_group
    parametrize_group = _parametrize_group
    _parametrize_group += 1

    outputs = []
    for param_values in zip(*value_sets):

        # Optional cls parameter in case a parameter is defined inside a
        # class scope.
        def fixture_func(*_cls, request):
            return request.param

        fixture_func.parametrize_group = parametrize_group
        fixture_func.parametrize_values = param_values
        fixture_func.parametrize_ids = ids
        outputs.append(pytest.fixture(fixture_func))

    return outputs


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
    if used with tvm.testing.parameter), and the same return value
    will be passed to all tests that use it.  If the environment
    variable TVM_TEST_DISABLE_CACHE is set to a non-zero value, it
    will disable this feature and no caching will be performed.

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

    # Deliberately at function scope, so that caching can track how
    # many times the fixture has been used.  If used, the cache gets
    # cleared after the fixture is no longer needed.
    scope = "function"

    def wraps(func):
        if cache_return_value:
            func = _fixture_cache(func)
        func = pytest.fixture(func, scope=scope)
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
            (
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
        )


def _fixture_cache(func):
    cache = {}

    # Can't use += on a bound method's property.  Therefore, this is a
    # list rather than a variable so that it can be accessed from the
    # pytest_collection_modifyitems().
    num_tests_use_this_fixture = [0]

    num_times_fixture_used = 0

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
        except TypeError as e:
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
        if num_tests_use_this_fixture[0] == 0:
            raise RuntimeError(
                "Fixture use count is 0.  "
                "This can occur if tvm.testing.plugin isn't registered.  "
                "If using outside of the TVM test directory, "
                "please add `pytest_plugins = ['tvm.testing.plugin']` to your conftest.py"
            )

        try:
            cache_key = get_cache_key(*args, **kwargs)

            try:
                cached_value = cache[cache_key]
            except KeyError:
                cached_value = cache[cache_key] = func(*args, **kwargs)

            yield copy.deepcopy(
                cached_value,
                # allowed_class_list should be a list of classes that
                # are safe to copy using copy.deepcopy, but do not
                # implement __deepcopy__, __reduce__, or
                # __reduce_ex__.
                _DeepCopyAllowedClasses(allowed_class_list=[]),
            )

        finally:
            # Clear the cache once all tests that use a particular fixture
            # have completed.
            nonlocal num_times_fixture_used
            num_times_fixture_used += 1
            if num_times_fixture_used >= num_tests_use_this_fixture[0]:
                cache.clear()

    # Set in the pytest_collection_modifyitems(), by _count_num_fixture_uses
    wrapper.num_tests_use_this_fixture = num_tests_use_this_fixture

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
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    return major >= 8


def install_request_hook(depth: int) -> None:
    """Add a wrapper around urllib.request for CI tests"""
    if not IS_IN_CI:
        return

    # https://sphinx-gallery.github.io/stable/faq.html#why-is-file-not-defined-what-can-i-use
    base = None
    msg = ""
    try:
        base = __file__
        msg += f"found file {__file__}\n"
    except NameError:
        msg += f"no file\n"

    if base is None:
        hook_script_dir = Path.cwd().resolve()
        msg += "used path.cwd()\n"
    else:
        hook_script_dir = Path(base).resolve().parent
        msg += "used base()\n"

    msg += f"using depth {depth}\n"
    if depth <= 0:
        raise ValueError(f"depth less than 1 not supported, found: {depth}")

    # Go up the parent directories
    while depth > 0:
        msg += f"[depth={depth}] dir={hook_script_dir}\n"
        hook_script_dir = hook_script_dir.parent
        depth -= 1

    # Ensure the specified dir is valid
    hook_script_dir = hook_script_dir / "tests" / "scripts" / "request_hook"
    if not hook_script_dir.exists():
        raise RuntimeError(f"Directory {hook_script_dir} does not exist:\n{msg}")

    # Import the hook and start it up (it's not included here directly to avoid
    # keeping a database of URLs inside the tvm Python package
    sys.path.append(str(hook_script_dir))
    # This import is intentionally delayed since it should only happen in CI
    import request_hook  # pylint: disable=import-outside-toplevel

    request_hook.init()


def fetch_model_from_url(
    url: str,
    model_format: str,
    sha256: str,
) -> Tuple[tvm.ir.module.IRModule, dict]:
    """Testing function to fetch a model from a URL and return it as a Relay
    model. Downloaded files are cached for future re-use.

    Parameters
    ----------
    url : str
        The URL or list of URLs to try downloading the model from.

    model_format: str
        The file extension of the model format used.

    sha256 : str
        The sha256 hex hash to compare the downloaded model against.

    Returns
    -------
    (mod, params) : object
        The Relay representation of the downloaded model.
    """

    rel_path = f"model_{sha256}.{model_format}"
    file = tvm.contrib.download.download_testdata(url, rel_path, overwrite=False)

    # Check SHA-256 hash
    file_hash = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(2**24), b""):
            file_hash.update(block)

    if file_hash.hexdigest() != sha256:
        raise FileNotFoundError("SHA-256 hash for model does not match")

    tvmc_model = load_model(file, model_format)
    return tvmc_model.mod, tvmc_model.params


def _mark_parameterizations(*params, marker_fn, reason):
    """
    Mark tests with a nodeid parameters that exactly matches one in params.
    Useful for quickly marking tests as xfail when they have a large
    combination of parameters.
    """
    params = set(params)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(request, *args, **kwargs):
            if "[" in request.node.name and "]" in request.node.name:
                # Strip out the test name and the [ and ] brackets
                params_from_name = request.node.name[len(request.node.originalname) + 1 : -1]
                if params_from_name in params:
                    marker_fn(
                        reason=f"{marker_fn.__name__} on nodeid {request.node.nodeid}: " + reason
                    )

            return func(request, *args, **kwargs)

        return wrapper

    return decorator


def xfail_parameterizations(*xfail_params, reason):
    return _mark_parameterizations(*xfail_params, marker_fn=pytest.xfail, reason=reason)


def skip_parameterizations(*skip_params, reason):
    return _mark_parameterizations(*skip_params, marker_fn=pytest.skip, reason=reason)


def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


class CompareBeforeAfter:
    """Utility for comparing before/after of TIR transforms

    A standard framework for writing tests that take a TIR PrimFunc as
    input, apply a transformation, then either compare against an
    expected output or assert that the transformation raised an error.
    A test should subclass CompareBeforeAfter, defining class members
    `before`, `transform`, and `expected`.  CompareBeforeAfter will
    then use these members to define a test method and test fixture.

    `transform` may be one of the following.

    - An instance of `tvm.ir.transform.Pass`

    - A method that takes no arguments and returns a `tvm.ir.transform.Pass`

    - A pytest fixture that returns a `tvm.ir.transform.Pass`

    `before` may be any one of the following.

    - An instance of `tvm.tir.PrimFunc`.  This is allowed, but is not
      the preferred method, as any errors in constructing the
      `PrimFunc` occur while collecting the test, preventing any other
      tests in the same file from being run.

    - An TVMScript function, without the ``@T.prim_func`` decoration.
      The ``@T.prim_func`` decoration will be applied when running the
      test, rather than at module import.

    - A method that takes no arguments and returns a `tvm.tir.PrimFunc`

    - A pytest fixture that returns a `tvm.tir.PrimFunc`

    `expected` may be any one of the following.  The type of
    `expected` defines the test being performed.  If `expected`
    provides a `tvm.tir.PrimFunc`, the result of the transformation
    must match `expected`.  If `expected` is an exception, then the
    transformation must raise that exception type.

    - Any option supported for `before`.

    - The `Exception` class object, or a class object that inherits
      from `Exception`.

    - A method that takes no arguments and returns `Exception` or a
      class object that inherits from `Exception`.

    - A pytest fixture that returns `Exception` or an class object
      that inherits from `Exception`.

    Examples
    --------

    .. python::

        class TestRemoveIf(tvm.testing.CompareBeforeAfter):
            transform = tvm.tir.transform.Simplify()

            def before(A: T.Buffer[1, "int32"]):
                if True:
                    A[0] = 42
                else:
                    A[0] = 5

            def expected(A: T.Buffer[1, "int32"]):
                A[0] = 42

    """

    def __init_subclass__(cls):
        if hasattr(cls, "before"):
            cls.before = cls._normalize_before(cls.before)
        if hasattr(cls, "expected"):
            cls.expected = cls._normalize_expected(cls.expected)
        if hasattr(cls, "transform"):
            cls.transform = cls._normalize_transform(cls.transform)

    @classmethod
    def _normalize_ir_module(cls, func):
        if isinstance(func, tvm.tir.PrimFunc):

            def inner(self):
                # pylint: disable=unused-argument
                return func

        elif cls._is_method(func):

            def inner(self):
                # pylint: disable=unused-argument
                return func(self)

        elif inspect.isclass(func):

            def inner(self):
                # pylint: disable=unused-argument
                func_dict = {}
                for name, method in func.__dict__.items():
                    if name.startswith("_"):
                        pass
                    elif isinstance(method, tvm.ir.function.BaseFunc):
                        func_dict[name] = method
                    else:
                        source_code = "@T.prim_func\n" + textwrap.dedent(inspect.getsource(method))
                        prim_func = tvm.script.from_source(source_code)
                        func_dict[name] = prim_func
                return tvm.IRModule(func_dict)

        else:

            def inner(self):
                # pylint: disable=unused-argument
                source_code = "@T.prim_func\n" + textwrap.dedent(inspect.getsource(func))
                return tvm.script.from_source(source_code)

        return pytest.fixture(inner)

    @classmethod
    def _normalize_before(cls, func):
        if hasattr(func, "_pytestfixturefunction"):
            return func
        else:
            return cls._normalize_ir_module(func)

    @classmethod
    def _normalize_expected(cls, func):
        if hasattr(func, "_pytestfixturefunction"):
            return func

        elif inspect.isclass(func) and issubclass(func, Exception):

            def inner(self):
                # pylint: disable=unused-argument
                return func

            return pytest.fixture(inner)

        else:
            return cls._normalize_ir_module(func)

    @classmethod
    def _normalize_transform(cls, transform):
        def apply(module_transform):
            def inner(obj):
                if isinstance(obj, tvm.IRModule):
                    return module_transform(obj)
                elif isinstance(obj, tvm.tir.PrimFunc):
                    mod = tvm.IRModule({"main": obj})
                    mod = module_transform(mod)
                    return mod["main"]
                else:
                    raise TypeError(f"Expected IRModule or PrimFunc, but received {type(obj)}")

            return inner

        if hasattr(transform, "_pytestfixturefunction"):

            if not hasattr(cls, "_transform_orig"):
                cls._transform_orig = transform

            def inner(self, _transform_orig):
                # pylint: disable=unused-argument
                return apply(_transform_orig)

        elif isinstance(transform, tvm.ir.transform.Pass):

            def inner(self):
                # pylint: disable=unused-argument
                return apply(transform)

        elif cls._is_method(transform):

            def inner(self):
                # pylint: disable=unused-argument
                return apply(transform(self))

        else:

            raise TypeError(
                "Expected transform to be a tvm.ir.transform.Pass, or a method returning a Pass"
            )

        return pytest.fixture(inner)

    @staticmethod
    def _is_method(func):
        sig = inspect.signature(func)
        return "self" in sig.parameters

    def test_compare(self, before, expected, transform):
        """Unit test to compare the expected TIR PrimFunc to actual"""

        def pprint(name, obj):
            script = obj.script()
            if isinstance(obj, tvm.IRModule):
                return script.replace("class Module", f"class {name}")
            else:
                return script.replace("def func", f"def {name}")

        if inspect.isclass(expected) and issubclass(expected, Exception):
            with pytest.raises(expected):
                after = transform(before)

                # This portion through pytest.fail isn't strictly
                # necessary, but gives a better error message that
                # includes the before/after.
                before_str = pprint("before", before)
                after_str = pprint("after", after)

                pytest.fail(
                    msg=(
                        f"Expected {expected.__name__} to be raised from transformation, "
                        f"instead received TIR\n:{before_str}\n{after_str}"
                    )
                )

        elif isinstance(expected, (tvm.tir.PrimFunc, tvm.ir.IRModule)):
            after = transform(before)

            try:
                tvm.ir.assert_structural_equal(after, expected)
            except ValueError as err:
                before_str = pprint("before", before)
                after_str = pprint("after", after)
                expected_str = pprint("expected", expected)
                raise ValueError(
                    f"TIR after transformation did not match expected:\n"
                    f"{before_str}\n{after_str}\n{expected_str}"
                ) from err

        else:
            raise TypeError(
                f"tvm.testing.CompareBeforeAfter requires the `expected` fixture "
                f"to return either `Exception`, an `Exception` subclass, "
                f"or an instance of `tvm.tir.PrimFunc`.  "
                f"Instead, received {type(expected)}."
            )
