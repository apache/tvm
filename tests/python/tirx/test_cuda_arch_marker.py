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
"""Unit tests for TIRx's exact CUDA architecture marker."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def _tirx_conftest(pytestconfig):
    expected = Path(__file__).with_name("conftest.py").resolve()
    for plugin in pytestconfig.pluginmanager.get_plugins():
        plugin_path = getattr(plugin, "__file__", None)
        if plugin_path is not None and Path(plugin_path).resolve() == expected:
            return plugin
    raise AssertionError(f"TIRx conftest plugin {expected} is not loaded")


def _item(marker):
    return SimpleNamespace(get_closest_marker=lambda name: marker if name == "cuda_arch" else None)


@pytest.mark.parametrize("arches", [(), ("sm_100a",), ("sm_100a", "sm_100a")])
def test_cuda_arch_pool_accepts_empty_or_homogeneous(arches, pytestconfig, monkeypatch):
    plugin = _tirx_conftest(pytestconfig)
    monkeypatch.setattr(plugin, "_visible_cuda_archs", lambda: arches)

    expected = None if not arches else "sm_100a"
    assert plugin._require_homogeneous_cuda_archs() == expected


def test_cuda_arch_pool_rejects_mixed_architectures(pytestconfig, monkeypatch):
    plugin = _tirx_conftest(pytestconfig)
    monkeypatch.setattr(plugin, "_visible_cuda_archs", lambda: ("sm_100a", "sm_107a"))

    with pytest.raises(
        pytest.UsageError, match=r"mixed-architecture.*cuda:0=sm_100a.*cuda:1=sm_107a"
    ):
        plugin._require_homogeneous_cuda_archs()


def test_cuda_arch_pool_rejects_unknown_architecture(pytestconfig, monkeypatch):
    plugin = _tirx_conftest(pytestconfig)
    monkeypatch.setattr(plugin, "_visible_cuda_archs", lambda: ("sm_100a", None))

    with pytest.raises(pytest.UsageError, match="could not determine"):
        plugin._require_homogeneous_cuda_archs()


def test_cuda_arch_marker_accepts_current_device(pytestconfig, monkeypatch):
    plugin = _tirx_conftest(pytestconfig)
    marker = pytest.mark.cuda_arch("sm_100a", "sm_103a", device="current").mark
    monkeypatch.setattr(plugin, "_set_cuda_device_for_xdist_worker", lambda: 7)
    monkeypatch.setattr(plugin.env, "cuda_arch", lambda device_id: "sm_103a")

    plugin.pytest_runtest_setup(_item(marker))


def test_cuda_arch_marker_skips_mismatched_device(pytestconfig, monkeypatch):
    plugin = _tirx_conftest(pytestconfig)
    marker = pytest.mark.cuda_arch("sm_100a", device="current").mark
    monkeypatch.setattr(plugin, "_set_cuda_device_for_xdist_worker", lambda: 7)
    monkeypatch.setattr(plugin.env, "cuda_arch", lambda device_id: "sm_103a")

    with pytest.raises(pytest.skip.Exception, match="actual architecture is sm_103a"):
        plugin.pytest_runtest_setup(_item(marker))


@pytest.mark.parametrize(
    "marker",
    [
        pytest.mark.cuda_arch().mark,
        pytest.mark.cuda_arch("SM_100a").mark,
        pytest.mark.cuda_arch("sm_100x").mark,
        pytest.mark.cuda_arch("sm_100a", device="worker").mark,
        pytest.mark.cuda_arch("sm_100a", unsupported=True).mark,
    ],
)
def test_cuda_arch_marker_rejects_invalid_usage(marker, pytestconfig, monkeypatch):
    plugin = _tirx_conftest(pytestconfig)
    monkeypatch.setattr(plugin, "_set_cuda_device_for_xdist_worker", lambda: 0)

    with pytest.raises(pytest.UsageError):
        plugin.pytest_runtest_setup(_item(marker))
