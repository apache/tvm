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
"""Configure pytest for TVM's Python test suite."""

import os

import _pytest


def pytest_collection_modifyitems(items):
    """Maintain the ordering and cache bookkeeping required by TVM fixtures."""
    _count_num_fixture_uses(items)
    _remove_global_fixture_definitions(items)
    _sort_tests(items)


def _count_num_fixture_uses(items):
    for item in items:
        is_skipped = item.get_closest_marker("skip") or any(
            mark.args[0] for mark in item.iter_markers("skipif")
        )
        if is_skipped:
            continue

        for fixturedefs in item._fixtureinfo.name2fixturedefs.values():
            fixturedef = fixturedefs[-1]
            if hasattr(fixturedef.func, "num_tests_use_this_fixture"):
                fixturedef.func.num_tests_use_this_fixture[0] += 1


def _remove_global_fixture_definitions(items):
    modules = {item.module for item in items}
    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, "_pytestfixturefunction") and isinstance(
                obj._pytestfixturefunction, _pytest.fixtures.FixtureFunctionMarker
            ):
                delattr(module, name)


def _sort_tests(items):
    def sort_key(item):
        filename, lineno, test_name = item.location
        return filename, lineno, test_name.split("[")[0]

    items.sort(key=sort_key)


def pytest_sessionstart():
    if os.getenv("CI", "") == "true":
        from request_hook import init  # pylint: disable=import-outside-toplevel

        init()
