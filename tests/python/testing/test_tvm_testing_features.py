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
# ruff: noqa: F401, RUF012

import os
import sys

import pytest

import tvm.testing

# This file tests features in tvm.testing, such as verifying that
# cached fixtures are run an appropriate number of times.  As a
# result, the order of the tests is important.  Use of --last-failed
# or --failed-first while debugging this file is not advised.  If
# these tests are distributed/parallelized using pytest-xdist or
# similar, all tests in this file should run sequentially on the same
# node.  (See https://stackoverflow.com/a/59504228)


class TestParameter:
    param1_vals = [1, 2, 3]
    param2_vals = ["a", "b", "c"]

    independent_usages = 0
    param1 = tvm.testing.parameter(*param1_vals)
    param2 = tvm.testing.parameter(*param2_vals)

    def test_using_independent(self, param1, param2):
        type(self).independent_usages += 1

    def test_independent(self):
        assert self.independent_usages == len(self.param1_vals) * len(self.param2_vals)


class TestFixtureCaching:
    param1_vals = [1, 2, 3]
    param2_vals = ["a", "b", "c"]

    param1 = tvm.testing.parameter(*param1_vals)
    param2 = tvm.testing.parameter(*param2_vals)

    uncached_calls = 0
    cached_calls = 0

    @tvm.testing.fixture
    def uncached_fixture(self, param1):
        type(self).uncached_calls += 1
        return 2 * param1

    def test_use_uncached(self, param1, param2, uncached_fixture):
        assert 2 * param1 == uncached_fixture

    def test_uncached_count(self):
        assert self.uncached_calls == len(self.param1_vals) * len(self.param2_vals)

    @tvm.testing.fixture(cache_return_value=True)
    def cached_fixture(self, param1):
        type(self).cached_calls += 1
        return 3 * param1

    def test_use_cached(self, param1, param2, cached_fixture):
        assert 3 * param1 == cached_fixture

    def test_cached_count(self):
        cache_disabled = bool(int(os.environ.get("TVM_TEST_DISABLE_CACHE", "0")))
        if cache_disabled:
            assert self.cached_calls == len(self.param1_vals) * len(self.param2_vals)
        else:
            assert self.cached_calls == len(self.param1_vals)


class TestCachedFixtureIsCopy:
    param = tvm.testing.parameter(1, 2, 3, 4)

    @tvm.testing.fixture(cache_return_value=True)
    def cached_mutable_fixture(self):
        return {"val": 0}

    def test_modifies_fixture(self, param, cached_mutable_fixture):
        assert cached_mutable_fixture["val"] == 0

        # The tests should receive a copy of the fixture value.  If
        # the test receives the original and not a copy, then this
        # will cause the next parametrization to fail.
        cached_mutable_fixture["val"] = param


class TestBrokenFixture:
    # Tests that use a fixture that throws an exception fail, and are
    # marked as setup failures.  The tests themselves are never run.
    # This behavior should be the same whether or not the fixture
    # results are cached.

    num_uses_broken_uncached_fixture = 0
    num_uses_broken_cached_fixture = 0

    @tvm.testing.fixture
    def broken_uncached_fixture(self):
        raise RuntimeError("Intentionally broken fixture")

    @pytest.mark.xfail(True, reason="Broken fixtures should result in a failing setup", strict=True)
    def test_uses_broken_uncached_fixture(self, broken_uncached_fixture):
        type(self).num_uses_broken_fixture += 1

    def test_num_uses_uncached(self):
        assert self.num_uses_broken_uncached_fixture == 0

    @tvm.testing.fixture(cache_return_value=True)
    def broken_cached_fixture(self):
        raise RuntimeError("Intentionally broken fixture")

    @pytest.mark.xfail(True, reason="Broken fixtures should result in a failing setup", strict=True)
    def test_uses_broken_cached_fixture(self, broken_cached_fixture):
        type(self).num_uses_broken_cached_fixture += 1

    def test_num_uses_cached(self):
        assert self.num_uses_broken_cached_fixture == 0


@pytest.mark.skipif(
    bool(int(os.environ.get("TVM_TEST_DISABLE_CACHE", "0"))),
    reason="Cannot test cache behavior while caching is disabled",
)
class TestCacheableTypes:
    class EmptyClass:
        pass

    @tvm.testing.fixture(cache_return_value=True)
    def uncacheable_fixture(self):
        return self.EmptyClass()

    def test_uses_uncacheable(self, request):
        # Normally the num_tests_use_this_fixture would be set before
        # anything runs.  For this test case only, because we are
        # delaying the use of the fixture, we need to manually
        # increment it.
        self.uncacheable_fixture.num_tests_use_this_fixture[0] += 1
        with pytest.raises(TypeError):
            request.getfixturevalue("uncacheable_fixture")

    class ImplementsReduce:
        def __reduce__(self):
            return super().__reduce__()

    @tvm.testing.fixture(cache_return_value=True)
    def fixture_with_reduce(self):
        return self.ImplementsReduce()

    def test_uses_reduce(self, fixture_with_reduce):
        pass

    class ImplementsDeepcopy:
        def __deepcopy__(self, memo):
            return type(self)()

    @tvm.testing.fixture(cache_return_value=True)
    def fixture_with_deepcopy(self):
        return self.ImplementsDeepcopy()

    def test_uses_deepcopy(self, fixture_with_deepcopy):
        pass


class TestPytestCache:
    param = tvm.testing.parameter(1, 2, 3)

    @pytest.fixture(scope="class")
    def cached_fixture(self, param):
        return param * param

    def test_uses_cached_fixture(self, param, cached_fixture):
        assert cached_fixture == param * param


if __name__ == "__main__":
    tvm.testing.main()
