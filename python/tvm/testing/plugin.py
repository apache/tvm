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
# pylint: disable=unused-argument

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

import _pytest


def pytest_collection_modifyitems(config, items):
    """Called after all tests are chosen, currently used for bookkeeping."""
    # pylint: disable=unused-argument
    _count_num_fixture_uses(items)
    _remove_global_fixture_definitions(items)
    _sort_tests(items)


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
