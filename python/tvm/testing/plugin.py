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

"""Pytest plugin for using tvm testing extensions.

TVM provides utilities for testing across all supported targets, and
to more easily parametrize across many inputs.  For more information
on usage of these features, see documentation in the tvm.testing
module.

These are enabled by default in all pytests provided by tvm, but may
be useful externally for one-off testing.  To enable, add the
following line to the test script, or to the conftest.py in the same
directory as the test scripts.

     pytest_plugins = ['tvm.testing.plugin']

"""

import collections

import pytest
import _pytest

import tvm
from tvm.testing import utils


MARKERS = {
    "gpu": "mark a test as requiring a gpu",
    "tensorcore": "mark a test as requiring a tensorcore",
    "cuda": "mark a test as requiring cuda",
    "opencl": "mark a test as requiring opencl",
    "rocm": "mark a test as requiring rocm",
    "vulkan": "mark a test as requiring vulkan",
    "metal": "mark a test as requiring metal",
    "llvm": "mark a test as requiring llvm",
    "ethosn": "mark a test as requiring ethosn",
}


def pytest_configure(config):
    """Runs at pytest configure time, defines marks to be used later."""

    for markername, desc in MARKERS.items():
        config.addinivalue_line("markers", "{}: {}".format(markername, desc))

    print("enabled targets:", "; ".join(map(lambda x: x[0], utils.enabled_targets())))
    print("pytest marker:", config.option.markexpr)


def pytest_generate_tests(metafunc):
    """Called once per unit test, modifies/parametrizes it as needed."""
    _parametrize_correlated_parameters(metafunc)
    _auto_parametrize_target(metafunc)


def pytest_collection_modifyitems(config, items):
    """Called after all tests are chosen, currently used for bookkeeping."""
    # pylint: disable=unused-argument
    _count_num_fixture_uses(items)
    _remove_global_fixture_definitions(items)


@pytest.fixture
def dev(target):
    """Give access to the device to tests that need it."""
    return tvm.device(target)


def pytest_sessionfinish(session, exitstatus):
    # Don't exit with an error if we select a subset of tests that doesn't
    # include anything
    if session.config.option.markexpr != "":
        if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
            session.exitstatus = pytest.ExitCode.OK


def _auto_parametrize_target(metafunc):
    """Automatically applies parametrize_targets

    Used if a test function uses the "target" fixture, but isn't
    already marked with @tvm.testing.parametrize_targets.  Intended
    for use in the pytest_generate_tests() handler of a conftest.py
    file.

    """

    def update_parametrize_target_arg(
        argnames,
        argvalues,
        *args,
        **kwargs,
    ):
        args = [arg.strip() for arg in argnames.split(",") if arg.strip()]
        if "target" in args:
            target_i = args.index("target")

            new_argvalues = []
            for argvalue in argvalues:

                if isinstance(argvalue, _pytest.mark.structures.ParameterSet):
                    # The parametrized value is already a
                    # pytest.param, so track any marks already
                    # defined.
                    param_set = argvalue.values
                    target = param_set[target_i]
                    additional_marks = argvalue.marks
                elif len(args) == 1:
                    # Single value parametrization, argvalue is a list of values.
                    target = argvalue
                    param_set = (target,)
                    additional_marks = []
                else:
                    # Multiple correlated parameters, argvalue is a list of tuple of values.
                    param_set = argvalue
                    target = param_set[target_i]
                    additional_marks = []

                new_argvalues.append(
                    pytest.param(
                        *param_set, marks=_target_to_requirement(target) + additional_marks
                    )
                )

            try:
                argvalues[:] = new_argvalues
            except TypeError as err:
                pyfunc = metafunc.definition.function
                filename = pyfunc.__code__.co_filename
                line_number = pyfunc.__code__.co_firstlineno
                msg = (
                    f"Unit test {metafunc.function.__name__} ({filename}:{line_number}) "
                    "is parametrized using a tuple of parameters instead of a list "
                    "of parameters."
                )
                raise TypeError(msg) from err

    if "target" in metafunc.fixturenames:
        # Update any explicit use of @pytest.mark.parmaetrize to
        # parametrize over targets.  This adds the appropriate
        # @tvm.testing.requires_* markers for each target.
        for mark in metafunc.definition.iter_markers("parametrize"):
            update_parametrize_target_arg(*mark.args, **mark.kwargs)

        # Check if any explicit parametrizations exist, and apply one
        # if they do not.  If the function is marked with either
        # excluded or known failing targets, use these to determine
        # the targets to be used.
        parametrized_args = [
            arg.strip()
            for mark in metafunc.definition.iter_markers("parametrize")
            for arg in mark.args[0].split(",")
        ]
        if "target" not in parametrized_args:
            excluded_targets = getattr(metafunc.function, "tvm_excluded_targets", [])
            xfail_targets = getattr(metafunc.function, "tvm_known_failing_targets", [])
            metafunc.parametrize(
                "target",
                _pytest_target_params(None, excluded_targets, xfail_targets),
                scope="session",
            )


def _count_num_fixture_uses(items):
    # Helper function, counts the number of tests that use each cached
    # fixture.  Should be called from pytest_collection_modifyitems().
    for item in items:
        is_skipped = item.get_closest_marker("skip") or any(
            mark.args[0] for mark in item.iter_markers("skipif")
        )
        if is_skipped:
            continue

        for fixturedefs in item._fixtureinfo.name2fixturedefs.values():
            # Only increment the active fixturedef, in a name has been overridden.
            fixturedef = fixturedefs[-1]
            if hasattr(fixturedef.func, "num_tests_use_this_fixture"):
                fixturedef.func.num_tests_use_this_fixture[0] += 1


def _remove_global_fixture_definitions(items):
    # Helper function, removes fixture definitions from the global
    # variables of the modules they were defined in.  This is intended
    # to improve readability of error messages by giving a NameError
    # if a test function accesses a pytest fixture but doesn't include
    # it as an argument.  Should be called from
    # pytest_collection_modifyitems().

    modules = set(item.module for item in items)

    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, "_pytestfixturefunction") and isinstance(
                obj._pytestfixturefunction, _pytest.fixtures.FixtureFunctionMarker
            ):
                delattr(module, name)


def _pytest_target_params(targets, excluded_targets=None, xfail_targets=None):
    # Include unrunnable targets here.  They get skipped by the
    # pytest.mark.skipif in _target_to_requirement(), showing up as
    # skipped tests instead of being hidden entirely.
    if targets is None:
        if excluded_targets is None:
            excluded_targets = set()

        if xfail_targets is None:
            xfail_targets = set()

        target_marks = []
        for t in utils._get_targets():
            # Excluded targets aren't included in the params at all.
            if t["target_kind"] not in excluded_targets:

                # Known failing targets are included, but are marked
                # as expected to fail.
                extra_marks = []
                if t["target_kind"] in xfail_targets:
                    extra_marks.append(
                        pytest.mark.xfail(
                            reason='Known failing test for target "{}"'.format(t["target_kind"])
                        )
                    )

                target_marks.append((t["target"], extra_marks))

    else:
        target_marks = [(target, []) for target in targets]

    return [
        pytest.param(target, marks=_target_to_requirement(target) + extra_marks)
        for target, extra_marks in target_marks
    ]


def _target_to_requirement(target):
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # mapping from target to decorator
    if target.kind.name == "cuda" and "cudnn" in target.attrs.get("libs", []):
        return utils.requires_cudnn()
    if target.kind.name == "cuda":
        return utils.requires_cuda()
    if target.kind.name == "rocm":
        return utils.requires_rocm()
    if target.kind.name == "vulkan":
        return utils.requires_vulkan()
    if target.kind.name == "nvptx":
        return utils.requires_nvptx()
    if target.kind.name == "metal":
        return utils.requires_metal()
    if target.kind.name == "opencl":
        return utils.requires_opencl()
    if target.kind.name == "llvm":
        return utils.requires_llvm()
    return []


def _parametrize_correlated_parameters(metafunc):
    parametrize_needed = collections.defaultdict(list)

    for name, fixturedefs in metafunc.definition._fixtureinfo.name2fixturedefs.items():
        fixturedef = fixturedefs[-1]
        if hasattr(fixturedef.func, "parametrize_group") and hasattr(
            fixturedef.func, "parametrize_values"
        ):
            group = fixturedef.func.parametrize_group
            values = fixturedef.func.parametrize_values
            parametrize_needed[group].append((name, values))

    for parametrize_group in parametrize_needed.values():
        if len(parametrize_group) == 1:
            name, values = parametrize_group[0]
            metafunc.parametrize(name, values, indirect=True)
        else:
            names = ",".join(name for name, values in parametrize_group)
            value_sets = zip(*[values for name, values in parametrize_group])
            metafunc.parametrize(names, value_sets, indirect=True)
