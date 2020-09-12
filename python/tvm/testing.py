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
""" TVM testing utilities

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
import logging
import os
import pytest
import numpy as np
import tvm
import tvm.arith
import tvm.tir
import tvm.te
import tvm._ffi
from tvm.contrib import nvcc


def assert_allclose(actual, desired, rtol=1e-7, atol=1e-7):
    """Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `actual` and `desired` are not interchangable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """
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
        grad_norm = np.sqrt(np.sum(ngrad ** 2))

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
    res = ana.simplify(lhs - rhs)
    equal = isinstance(res, tvm.tir.IntImm) and res.value == 0
    if not equal:
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
        return args[0].asnumpy()

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


def _get_targets():
    target_str = os.environ.get("TVM_TEST_TARGETS", "")
    if len(target_str) == 0:
        target_str = DEFAULT_TEST_TARGETS
    targets = set()
    for dev in target_str.split(";"):
        if len(dev) == 0:
            continue
        target_kind = dev.split()[0]
        if tvm.runtime.enabled(target_kind) and tvm.context(target_kind, 0).exist:
            targets.add(dev)
    if len(targets) == 0:
        logging.warning(
            "None of the following targets are supported by this build of TVM: %s."
            " Try setting TVM_TEST_TARGETS to a supported target. Defaulting to llvm.",
            target_str,
        )
        return {"llvm"}
    return targets


DEFAULT_TEST_TARGETS = (
    "llvm;cuda;opencl;metal;rocm;vulkan;nvptx;"
    "llvm -device=arm_cpu;opencl -device=mali,aocl_sw_emu"
)


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
    target_kind = target.split(" ")[
        0
    ]  # only check if device name is found, sometime there are extra flags
    return any([target_kind in test_target for test_target in _get_targets()])


def enabled_targets():
    """Get all enabled targets with associated contexts.

    In most cases, you should use :py:func:`tvm.testing.parametrize_targets` instead of
    this function.

    In this context, enabled means that TVM was built with support for this
    target and the target name appears in the TVM_TEST_TARGETS environment
    variable. If TVM_TEST_TARGETS is not set, it defaults to variable
    DEFAULT_TEST_TARGETS in this module.

    If you use this function in a test, you **must** decorate the test with
    :py:func:`tvm.testing.uses_gpu` (otherwise it will never be run on the gpu).

    Returns
    -------
    targets: list
        A list of pairs of all enabled devices and the associated context
    """
    return [(tgt, tvm.context(tgt)) for tgt in _get_targets()]


def _compose(args, decs):
    """Helper to apply multiple markers"""
    if len(args) > 0:
        f = args[0]
        for d in reversed(decs):
            f = d(f)
        return f
    return decs


def uses_gpu(*args):
    """Mark to differentiate tests that use the GPU is some capacity.

    These tests will be run on CPU-only test nodes and on test nodes with GPUS.
    To mark a test that must have a GPU present to run, use
    :py:func:`tvm.testing.requires_gpu`.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _uses_gpu = [pytest.mark.gpu]
    return _compose(args, _uses_gpu)


def requires_gpu(*args):
    """Mark a test as requiring a GPU to run.

    Tests with this mark will not be run unless a gpu is present.

    Parameters
    ----------
    f : function
        Function to mark
    """
    _requires_gpu = [
        pytest.mark.skipif(not tvm.gpu().exist, reason="No GPU present"),
        *uses_gpu(),
    ]
    return _compose(args, _requires_gpu)


def requires_cuda(*args):
    """Mark a test as requiring the CUDA runtime.

    This also marks the test as requiring a gpu.

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
            not tvm.gpu().exist or not nvcc.have_tensorcore(tvm.gpu(0).compute_version),
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


def _target_to_requirement(target):
    # mapping from target to decorator
    if target.startswith("cuda"):
        return requires_cuda()
    if target.startswith("rocm"):
        return requires_rocm()
    if target.startswith("vulkan"):
        return requires_vulkan()
    if target.startswith("nvptx"):
        return [*requires_llvm(), *requires_gpu()]
    if target.startswith("metal"):
        return requires_metal()
    if target.startswith("opencl"):
        return requires_opencl()
    if target.startswith("llvm"):
        return requires_llvm()
    return []


def parametrize_targets(*args):
    """Parametrize a test over all enabled targets.

    Use this decorator when you want your test to be run over a variety of
    targets and devices (including cpu and gpu devices).

    Parameters
    ----------
    f : function
        Function to parametrize. Must be of the form `def test_xxxxxxxxx(target, ctx)`:,
        where `xxxxxxxxx` is any name.
    targets : list[str], optional
        Set of targets to run against. If not supplied,
        :py:func:`tvm.testing.enabled_targets` will be used.

    Example
    -------
    >>> @tvm.testing.parametrize
    >>> def test_mytest(target, ctx):
    >>>     ...  # do something

    Or

    >>> @tvm.testing.parametrize("llvm", "cuda")
    >>> def test_mytest(target, ctx):
    >>>     ...  # do something
    """

    def wrap(targets):
        def func(f):
            params = [
                pytest.param(target, tvm.context(target, 0), marks=_target_to_requirement(target))
                for target in targets
            ]
            return pytest.mark.parametrize("target,ctx", params)(f)

        return func

    if len(args) == 1 and callable(args[0]):
        targets = [t for t, _ in enabled_targets()]
        return wrap(targets)(args[0])
    return wrap(args)


tvm._ffi._init_api("testing", __name__)
