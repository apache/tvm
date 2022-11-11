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

import pytest
import _pytest

import tvm
from tvm.testing import utils

try:
    from xdist.scheduler.loadscope import LoadScopeScheduling

    HAVE_XDIST = True
except ImportError:
    HAVE_XDIST = False


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
    "hexagon": "mark a test as requiring hexagon",
    "corstone300": "mark a test as requiring Corstone300 FVP",
}


def pytest_configure(config):
    """Runs at pytest configure time, defines marks to be used later."""

    for feature in utils.Feature._all_features.values():
        feature._register_marker(config)

    print("enabled targets:", "; ".join(map(lambda x: x[0], utils.enabled_targets())))
    print("pytest marker:", config.option.markexpr)


def pytest_addoption(parser):
    """Add pytest options."""
    parser.addoption("--gtest_args", action="store", default="")


def pytest_generate_tests(metafunc):
    """Called once per unit test, modifies/parametrizes it as needed."""
    _parametrize_correlated_parameters(metafunc)
    _auto_parametrize_target(metafunc)
    _add_target_specific_marks(metafunc)

    # Process gtest arguments
    option_value = metafunc.config.option.gtest_args
    if "gtest_args" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("gtest_args", [option_value])


def pytest_collection_modifyitems(config, items):
    """Called after all tests are chosen, currently used for bookkeeping."""
    # pylint: disable=unused-argument
    _count_num_fixture_uses(items)
    _remove_global_fixture_definitions(items)
    _sort_tests(items)


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

    if "target" in metafunc.fixturenames:
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

            # Add a parametrize marker instead of calling
            # metafunc.parametrize so that the parametrize rewriting
            # can still occur.
            mark = pytest.mark.parametrize(
                "target",
                [
                    t["target"]
                    for t in utils._get_targets()
                    if t["target_kind"] not in excluded_targets
                ],
                scope="session",
            )
            metafunc.definition.add_marker(mark)


def _add_target_specific_marks(metafunc):
    """Add any target-specific marks to parametrizations over target"""

    def update_parametrize_target_arg(
        mark,
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

                if mark in metafunc.definition.own_markers:
                    xfail_targets = getattr(metafunc.function, "tvm_known_failing_targets", [])
                    target_kind = target.split()[0] if isinstance(target, str) else target.kind.name
                    if target_kind in xfail_targets:
                        additional_marks.append(
                            pytest.mark.xfail(
                                reason=f'Known failing test for target "{target_kind}"'
                            )
                        )

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
            update_parametrize_target_arg(mark, *mark.args, **mark.kwargs)


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


def _sort_tests(items):
    """Sort tests by file/function.

    By default, pytest will sort tests to maximize the re-use of
    fixtures.  However, this assumes that all fixtures have an equal
    cost to generate, and no caches outside of those managed by
    pytest.  A tvm.testing.parameter is effectively free, while
    reference data for testing may be quite large.  Since most of the
    TVM fixtures are specific to a python function, sort the test
    ordering by python function, so that
    tvm.testing.utils._fixture_cache can be cleared sooner rather than
    later.

    Should be called from pytest_collection_modifyitems.

    """

    def sort_key(item):
        filename, lineno, test_name = item.location
        test_name = test_name.split("[")[0]
        return filename, lineno, test_name

    items.sort(key=sort_key)


def _target_to_requirement(target):
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # mapping from target to decorator
    if target.kind.name == "cuda" and "cudnn" in target.attrs.get("libs", []):
        return utils.requires_cudnn.marks()
    if target.kind.name == "cuda" and "cublas" in target.attrs.get("libs", []):
        return utils.requires_cublas.marks()
    if target.kind.name == "cuda":
        return utils.requires_cuda.marks()
    if target.kind.name == "rocm":
        return utils.requires_rocm.marks()
    if target.kind.name == "vulkan":
        return utils.requires_vulkan.marks()
    if target.kind.name == "nvptx":
        return utils.requires_nvptx.marks()
    if target.kind.name == "metal":
        return utils.requires_metal.marks()
    if target.kind.name == "opencl":
        return utils.requires_opencl.marks()
    if target.kind.name == "llvm":
        return utils.requires_llvm.marks()
    if target.kind.name == "hexagon":
        return utils.requires_hexagon.marks()

    return []


def _parametrize_correlated_parameters(metafunc):
    parametrize_needed = {}

    for name, fixturedefs in metafunc.definition._fixtureinfo.name2fixturedefs.items():
        fixturedef = fixturedefs[-1]
        if hasattr(fixturedef.func, "parametrize_group") and hasattr(
            fixturedef.func, "parametrize_values"
        ):
            group = fixturedef.func.parametrize_group
            values = fixturedef.func.parametrize_values
            ids = fixturedef.func.parametrize_ids
            if group in parametrize_needed:
                assert ids == parametrize_needed[group]["ids"]
            else:
                parametrize_needed[group] = {"ids": ids, "params": []}
            parametrize_needed[group]["params"].append((name, values))

    for parametrize_group in parametrize_needed.values():
        params = parametrize_group["params"]
        ids = parametrize_group["ids"]
        if len(params) == 1:
            name, values = params[0]
            metafunc.parametrize(name, values, indirect=True, ids=ids)
        else:
            names = ",".join(name for name, values in params)
            value_sets = zip(*[values for name, values in params])
            metafunc.parametrize(names, value_sets, indirect=True, ids=ids)


# pytest-xdist isn't required but is used in CI, so guard on its presence
if HAVE_XDIST:

    def pytest_xdist_make_scheduler(config, log):
        """
        Serialize certain tests for pytest-xdist that have inter-test
        dependencies
        """

        class TvmTestScheduler(LoadScopeScheduling):
            """
            Scheduler to serializer tests
            """

            def _split_scope(self, nodeid):
                """
                Returns a specific string for classes of nodeids
                """
                # NOTE: these tests contain inter-test dependencies and must be
                # serialized
                items = {
                    "test_tvm_testing_features": "functional-tests",
                    "tests/python/unittest/test_crt": "crt-tests",
                    "tests/python/driver/tvmc": "tvmc-tests",
                }

                for nodeid_pattern, suite_name in items.items():
                    if nodeid_pattern in nodeid:
                        return suite_name

                return nodeid

        return TvmTestScheduler(config, log)
