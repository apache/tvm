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
# ruff: noqa: RUF012

import os

import pytest

import tvm.testing


class TestParameter:
    param1_vals = [1, 2, 3]
    param2_vals = ["a", "b", "c"]

    param1 = tvm.testing.parameter(*param1_vals)
    param2 = tvm.testing.parameter(*param2_vals)

    def test_using_independent(self, param1, param2):
        assert param1 in self.param1_vals
        assert param2 in self.param2_vals


class TestFixtureCaching:
    param1_vals = [1, 2, 3]
    param2_vals = ["a", "b", "c"]

    param1 = tvm.testing.parameter(*param1_vals)
    param2 = tvm.testing.parameter(*param2_vals)

    @tvm.testing.fixture
    def uncached_fixture(self, param1):
        return 2 * param1

    def test_use_uncached(self, param1, param2, uncached_fixture):
        assert 2 * param1 == uncached_fixture

    @tvm.testing.fixture(cache_return_value=True)
    def cached_fixture(self, param1):
        return 3 * param1

    def test_use_cached(self, param1, param2, cached_fixture):
        assert 3 * param1 == cached_fixture


def test_fixture_cache_reuses_setup_and_returns_copies():
    setup_calls = []

    def setup(value):
        setup_calls.append(value)
        return {"value": value}

    cached_setup = tvm.testing.utils._fixture_cache(setup)
    first = cached_setup(1)
    first["value"] = 0

    assert cached_setup(1) == {"value": 1}
    assert cached_setup(2) == {"value": 2}
    assert setup_calls == [1, 2]


def test_request_hook_uses_explicit_path(monkeypatch, tmp_path):
    hook_script = tmp_path / "request_hook.py"
    hook_script.touch()
    hook_script = hook_script.resolve()
    loads = []
    initializations = []

    def load_hook(path):
        loads.append(path)
        return {"init": lambda: initializations.append(path)}

    monkeypatch.setattr(tvm.testing.utils, "IS_IN_CI", True)
    monkeypatch.setattr(
        tvm.testing.utils,
        "__file__",
        "/installed/site-packages/tvm/testing/utils.py",
    )
    monkeypatch.setattr(tvm.testing.utils.runpy, "run_path", load_hook)

    try:
        tvm.testing.utils.install_request_hook(hook_script)
        tvm.testing.utils.install_request_hook(hook_script)
    finally:
        tvm.testing.utils._REQUEST_HOOK_INITIALIZERS.pop(hook_script, None)

    assert loads == [str(hook_script)]
    assert initializations == [str(hook_script), str(hook_script)]


class TestBrokenFixture:
    # Tests that use a fixture that throws an exception fail, and are
    # marked as setup failures.  The tests themselves are never run.
    # This behavior should be the same whether or not the fixture
    # results are cached.

    @tvm.testing.fixture
    def broken_uncached_fixture(self):
        raise RuntimeError("Intentionally broken fixture")

    @pytest.mark.xfail(True, reason="Broken fixtures should result in a failing setup", strict=True)
    def test_uses_broken_uncached_fixture(self, broken_uncached_fixture):
        pass

    @tvm.testing.fixture(cache_return_value=True)
    def broken_cached_fixture(self):
        raise RuntimeError("Intentionally broken fixture")

    @pytest.mark.xfail(True, reason="Broken fixtures should result in a failing setup", strict=True)
    def test_uses_broken_cached_fixture(self, broken_cached_fixture):
        pass


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
