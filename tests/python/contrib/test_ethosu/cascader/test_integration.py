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

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.codegen import _create_cascader
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from tvm.contrib.ethosu.cascader import MemoryRegion, EthosuDeviceConfig, CascaderOptions

from .. import infra


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
    cascader_options = CascaderOptions(
        cascade_region=sram,
        max_proposals=64,
        stripe_factors=5,
        max_plan_size=10,
        always_copy_size=1024,
        enable_striping=False,
    )
    return _create_cascader(
        options=cascader_options,
        io_region=sram,
        constant_region=flash,
        working_regions=[sram],
        device_config=device_config,
    )


def _create_single_conv2d():
    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    conv1 = infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv1), conv1)
    return func


def test_check_compute_cycle_hint():
    """Check the "compute_cycle_hint" annotation remains in the lowering flow."""
    relay_function = _create_single_conv2d()
    mod = tvm.IRModule()
    mod["main"] = relay_function
    mod = relay.transform.InferType()(mod)
    tir_mod = _lower_to_tir(mod["main"], _ethos_u55_cascader())[0]
    primfunc = tir_mod["main"]

    npu_op = primfunc.body.body.body.seq[2]
    assert npu_op.attr_key == "pragma_compute_cycle_hint"
    assert npu_op.value == 320
