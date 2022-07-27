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
# pylint: disable=wrong-import-position,invalid-name

"""
Test the cascader in the compilation flow.
"""

import pytest

pytest.importorskip("ethosu.vela")

import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.codegen import _create_cascader
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from tvm.contrib.ethosu.cascader import MemoryRegion, EthosuDeviceConfig

from .. import infra as test_infra
from . import infra as cascader_test_infra


def _ethos_u55_cascader():
    sram = MemoryRegion(
        name="SRAM",
        size=10**6,
        read_bandwidth=16,
        write_bandwidth=16,
        read_latency=0,
        write_latency=0,
        burst_length=1,
    )
    flash = MemoryRegion(name="FLASH", size=10**7, read_bandwidth=4, write_bandwidth=4)

    device_config = EthosuDeviceConfig("ethos-u55-256")
    cascader_options = cascader_test_infra.make_options(
        cascade_region=sram,
        max_proposals=64,
        stripe_factors=4,
        max_plan_size=10,
        max_open_plans=8,
        max_closed_plans=32,
        always_copy_size=1024,
        disable_pareto_plans=False,
        disable_pareto_proposals=False,
        enable_striping=False,
    )
    return _create_cascader(
        options=cascader_options,
        io_region=sram,
        constant_region=flash,
        working_regions=[sram],
        device_config=device_config,
    )


def _compile_model(relay_function):
    mod = tvm.IRModule()
    mod["main"] = relay_function
    mod = relay.transform.InferType()(mod)
    tir_mod = _lower_to_tir(mod["main"], _ethos_u55_cascader())[0]
    return tir_mod["main"]


def _create_single_conv2d():
    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    conv1 = test_infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv1), conv1)
    return func


def _create_double_conv2d():
    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    conv1 = test_infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    conv2 = test_infra.make_ethosu_conv2d(conv1, 4, 4, (1, 3), (1, 1), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv2), conv2)
    return func


def _create_scalar_add():
    ifm = relay.var("x", shape=(1, 5, 4, 3), dtype="int8")
    ifm2 = relay.const(np.ones((1, 1, 1, 1)), dtype="int8")
    add = test_infra.make_ethosu_binary_elementwise(
        ifm, ifm2, ifm_channels=3, ifm2_channels=1, operator_type="ADD", ofm_dtype="int8"
    )
    func = relay.Function(relay.analysis.free_vars(add), add)
    return func


def test_single_conv_compute_cycles_hint():
    """
    Check the "compute_cycles_hint" annotation remains in the lowering flow
    for single convolution.
    """
    primfunc = _compile_model(_create_single_conv2d())
    ops = primfunc.body.body.seq
    compute_cycles_hints = [2944, 320]
    for op, compute_cycle_hint in zip(ops, compute_cycles_hints):
        assert op.attr_key == "pragma_compute_cycles_hint"
        assert op.value == compute_cycle_hint


def test_double_conv_compute_cycles_hint():
    """
    Check the "compute_cycles_hint" annotation remains in the lowering flow
    for double convolution.
    """
    primfunc = _compile_model(_create_double_conv2d())
    ops = primfunc.body.body.body.body.seq
    compute_cycles_hints = [2944, 1408, 320, 240]
    for op, compute_cycle_hint in zip(ops, compute_cycles_hints):
        assert op.attr_key == "pragma_compute_cycles_hint"
        assert op.value == compute_cycle_hint


def test_scalar_add_compute_cycles_hint():
    """
    Check the "compute_cycles_hint" annotation remains in the lowering flow
    for add with scalar values.
    """
    primfunc = _compile_model(_create_scalar_add())
    ops = primfunc.body.body.seq

    compute_cycles_hints = [16, 24]
    for op, compute_cycle_hint in zip(ops, compute_cycles_hints):
        assert op.attr_key == "pragma_compute_cycles_hint"
        assert op.value == compute_cycle_hint
