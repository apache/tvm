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


class TestTargetAutoParametrization:
    targets_used = []
    devices_used = []
    enabled_targets = [target for target, dev in tvm.testing.enabled_targets()]
    enabled_devices = [dev for target, dev in tvm.testing.enabled_targets()]

    def test_target_parametrization(self, target):
        assert target in self.enabled_targets
        self.targets_used.append(target)

    def test_device_parametrization(self, dev):
        assert dev in self.enabled_devices
        self.devices_used.append(dev)

    def test_all_targets_used(self):
        assert self.targets_used == self.enabled_targets
        assert self.devices_used == self.enabled_devices

    targets_with_explicit_list = []

    @tvm.testing.parametrize_targets("llvm")
    def test_explicit_list(self, target):
        assert target == "llvm"
        self.targets_with_explicit_list.append(target)

    def test_no_repeats_in_explicit_list(self):
        assert self.targets_with_explicit_list == ["llvm"]

    targets_with_exclusion = []

    @tvm.testing.exclude_targets("llvm")
    def test_exclude_target(self, target):
        assert "llvm" not in target
        self.targets_with_exclusion.append(target)

    def test_all_nonexcluded_targets_ran(self):
        assert self.targets_with_exclusion == [
            target for target in self.enabled_targets if not target.startswith("llvm")
        ]

    run_targets_with_known_failure = []

    @tvm.testing.known_failing_targets("llvm")
    def test_known_failing_target(self, target):
        # This test runs for all targets, but intentionally fails for
        # llvm.  The behavior is working correctly if this test shows
        # up as an expected failure, xfail.
        self.run_targets_with_known_failure.append(target)
        assert "llvm" not in target

    def test_all_targets_ran(self):
        assert self.run_targets_with_known_failure == self.enabled_targets


class TestJointParameter:
    param1_vals = [1, 2, 3]
    param2_vals = ["a", "b", "c"]

    independent_usages = 0
    param1 = tvm.testing.parameter(*param1_vals)
    param2 = tvm.testing.parameter(*param2_vals)

    joint_usages = 0
    joint_param_vals = list(zip(param1_vals, param2_vals))
    joint_param1, joint_param2 = tvm.testing.parameters(*joint_param_vals)

    def test_using_independent(self, param1, param2):
        type(self).independent_usages += 1

    def test_independent(self):
        assert self.independent_usages == len(self.param1_vals) * len(self.param2_vals)

    def test_using_joint(self, joint_param1, joint_param2):
        type(self).joint_usages += 1
        assert (joint_param1, joint_param2) in self.joint_param_vals

    def test_joint(self):
        assert self.joint_usages == len(self.joint_param_vals)


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


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
