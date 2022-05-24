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
import logging
import os
import platform
import shutil
import sys
import time
import pickle
import pytest
import numpy as np
import tvm
import tvm.arith
import tvm.tir
import tvm.te
import tvm._ffi

from tvm.contrib import nvcc, cudnn
from tvm.error import TVMError
from tvm.relay.op.contrib.ethosn import ethosn_available
from tvm.relay.op.contrib import cmsisnn
from tvm.relay.op.contrib import vitis_ai


SKIP_SLOW_TESTS = os.getenv("SKIP_SLOW_TESTS", "").lower() in {"true", "1", "yes"}


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


def _get_targets(target_str=None):
    if target_str is None:
        target_str = os.environ.get("TVM_TEST_TARGETS", "")
        # Use dict instead of set for de-duplication so that the
        # targets stay in the order specified.
        target_names = list({t.strip(): None for t in target_str.split(";") if t.strip()})

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
                target_str,
            )
            return _get_targets("llvm")

        raise TVMError(
            "None of the following targets are supported by this build of TVM: %s."
            " Try setting TVM_TEST_TARGETS to a supported target."
            " Cannot default to llvm, as it is not enabled." % target_str
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


def _compose(args, decs):
    """Helper to apply multiple markers"""
    if len(args) > 0:
        f = args[0]
        for d in reversed(decs):
            f = d(f)
        return f
    return decs


def slow(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if SKIP_SLOW_TESTS:
            pytest.skip("Skipping slow test since RUN_SLOW_TESTS environment variables is 'true'")
        else:
            fn(*args, **kwargs)

    return wrapper


def uses_gpu(*args):
    """Mark to differentiate tests that use the GPU in some capacity.

    These tests will be run on CPU-only test nodes and on test nodes with GPUs.
    To mark a test that must have a GPU present to run, use
    :py:func:`tvm.testing.requires_gpu`.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _uses_gpu = [pytest.mark.gpu]
    return _compose(args, _uses_gpu)


def requires_x86(*args):
    """Mark a test as requiring the x86 Architecture to run.

    Tests with this mark will not be run unless on an x86 platform.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_x86 = [
        pytest.mark.skipif(platform.machine() != "x86_64", reason="x86 Architecture Required"),
    ]
    return _compose(args, _requires_x86)


def requires_gpu(*args):
    """Mark a test as requiring a GPU to run.

    Tests with this mark will not be run unless a gpu is present.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_gpu = [
        pytest.mark.skipif(
            not tvm.cuda().exist
            and not tvm.rocm().exist
            and not tvm.opencl().exist
            and not tvm.metal().exist
            and not tvm.vulkan().exist,
            reason="No GPU present",
        ),
        *uses_gpu(),
    ]
    return _compose(args, _requires_gpu)


def requires_cuda(*args):
    """Mark a test as requiring the CUDA runtime.

    This also marks the test as requiring a cuda gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_cuda = [
        pytest.mark.cuda,
        pytest.mark.skipif(not device_enabled("cuda"), reason="CUDA support not enabled"),
        *requires_gpu(),
    ]
    return _compose(args, _requires_cuda)


def requires_cudnn(*args):
    """Mark a test as requiring the cuDNN library.

    This also marks the test as requiring a cuda gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """

    requirements = [
        pytest.mark.skipif(
            not cudnn.exists(), reason="cuDNN library not enabled, or not installed"
        ),
        *requires_cuda(),
    ]
    return _compose(args, requirements)


def requires_cublas(*args):
    """Mark a test as requiring the cuBLAS library.

    This also marks the test as requiring a cuda gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """

    requirements = [
        pytest.mark.skipif(
            tvm.get_global_func("tvm.contrib.cublas.matmul", True),
            reason="cuDNN library not enabled",
        ),
        *requires_cuda(),
    ]
    return _compose(args, requirements)


def requires_nvptx(*args):
    """Mark a test as requiring the NVPTX compilation on the CUDA runtime

    This also marks the test as requiring a cuda gpu, and requiring
    LLVM support.

    Parameters
    ----------
    f : function
        Function to mark

    """
    _requires_nvptx = [
        pytest.mark.skipif(not device_enabled("nvptx"), reason="NVPTX support not enabled"),
        *requires_llvm(),
        *requires_gpu(),
    ]
    return _compose(args, _requires_nvptx)


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
        *requires_cuda(),
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


def requires_cudagraph(*args):
    """Mark a test as requiring the CUDA Graph Feature

    This also marks the test as requiring cuda

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_cudagraph = [
        pytest.mark.skipif(
            not nvcc.have_cudagraph(), reason="CUDA Graph is not supported in this environment"
        ),
        *requires_cuda(),
    ]
    return _compose(args, _requires_cudagraph)


def requires_opencl(*args):
    """Mark a test as requiring the OpenCL runtime.

    This also marks the test as requiring a gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_opencl = [
        pytest.mark.opencl,
        pytest.mark.skipif(not device_enabled("opencl"), reason="OpenCL support not enabled"),
        *requires_gpu(),
    ]
    return _compose(args, _requires_opencl)


def requires_corstone300(*args):
    """Mark a test as requiring the corstone300 FVP

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_corstone300 = [
        pytest.mark.corstone300,
        pytest.mark.skipif(
            shutil.which("arm-none-eabi-gcc") is None, reason="ARM embedded toolchain unavailable"
        ),
    ]
    return _compose(args, _requires_corstone300)


def requires_rocm(*args):
    """Mark a test as requiring the rocm runtime.

    This also marks the test as requiring a gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_rocm = [
        pytest.mark.rocm,
        pytest.mark.skipif(not device_enabled("rocm"), reason="rocm support not enabled"),
        *requires_gpu(),
    ]
    return _compose(args, _requires_rocm)


def requires_metal(*args):
    """Mark a test as requiring the metal runtime.

    This also marks the test as requiring a gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_metal = [
        pytest.mark.metal,
        pytest.mark.skipif(not device_enabled("metal"), reason="metal support not enabled"),
        *requires_gpu(),
    ]
    return _compose(args, _requires_metal)


def requires_vulkan(*args):
    """Mark a test as requiring the vulkan runtime.

    This also marks the test as requiring a gpu.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_vulkan = [
        pytest.mark.vulkan,
        pytest.mark.skipif(not device_enabled("vulkan"), reason="vulkan support not enabled"),
        *requires_gpu(),
    ]
    return _compose(args, _requires_vulkan)


def requires_tensorcore(*args):
    """Mark a test as requiring a tensorcore to run.

    Tests with this mark will not be run unless a tensorcore is present.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_tensorcore = [
        pytest.mark.tensorcore,
        pytest.mark.skipif(
            not tvm.cuda().exist or not nvcc.have_tensorcore(tvm.cuda(0).compute_version),
            reason="No tensorcore present",
        ),
        *requires_gpu(),
    ]
    return _compose(args, _requires_tensorcore)


def requires_llvm(*args):
    """Mark a test as requiring llvm to run.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_llvm = [
        pytest.mark.llvm,
        pytest.mark.skipif(not device_enabled("llvm"), reason="LLVM support not enabled"),
    ]
    return _compose(args, _requires_llvm)


def requires_micro(*args):
    """Mark a test as requiring microTVM to run.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_micro = [
        pytest.mark.skipif(
            tvm.support.libinfo().get("USE_MICRO", "OFF") != "ON",
            reason="MicroTVM support not enabled. Set USE_MICRO=ON in config.cmake to enable.",
        )
    ]
    return _compose(args, _requires_micro)


def requires_rpc(*args):
    """Mark a test as requiring rpc to run.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_rpc = [
        pytest.mark.skipif(
            tvm.support.libinfo().get("USE_RPC", "OFF") != "ON",
            reason="RPC support not enabled. Set USE_RPC=ON in config.cmake to enable.",
        )
    ]
    return _compose(args, _requires_rpc)


def requires_ethosn(*args):
    """Mark a test as requiring Arm(R) Ethos(TM)-N to run.

    Parameters
    ----------
    f : function
        Function to mark
    """
    marks = [
        pytest.mark.ethosn,
        pytest.mark.skipif(
            not ethosn_available(),
            reason=(
                "Arm(R) Ethos(TM)-N support not enabled.  "
                "Set USE_ETHOSN=ON in config.cmake to enable, "
                "and ensure that hardware support is present."
            ),
        ),
    ]
    return _compose(args, marks)


def requires_hexagon(*args):
    """Mark a test as requiring Hexagon to run.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_hexagon = [
        pytest.mark.hexagon,
        pytest.mark.skipif(not device_enabled("hexagon"), reason="Hexagon support not enabled"),
        *requires_llvm(),
        pytest.mark.skipif(
            tvm.target.codegen.llvm_version_major() < 7, reason="Hexagon requires LLVM 7 or later"
        ),
    ]
    return _compose(args, _requires_hexagon)


def requires_cmsisnn(*args):
    """Mark a test as requiring the CMSIS NN library.

    Parameters
    ----------
    f : function
        Function to mark
    """

    requirements = [pytest.mark.skipif(not cmsisnn.enabled(), reason="CMSIS NN not enabled")]
    return _compose(args, requirements)


def requires_vitis_ai(*args):
    """Mark a test as requiring Vitis AI to run.

    Parameters
    ----------
    f : function
        Function to mark
    """

    requirements = [pytest.mark.skipif(not vitis_ai.enabled(), reason="Vitis AI not enabled")]
    return _compose(args, requirements)


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


def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))
