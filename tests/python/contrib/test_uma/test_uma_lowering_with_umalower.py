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
import pytest
import pathlib

import tvm
from tests.python.contrib.test_uma.test_uma_utils import _create_schedule, _generate_io_arrays
from tvm import topi
from apps.uma._template.passes import MyAiHwConv2dPass
import tvm.testing
from tvm import te
from tvm.relay.backend.contrib.uma.api.lower import UMALower
from tvm.relay.backend.contrib.uma.api.utils import PassPhase
from tvm.relay.backend.contrib.uma import uma_available


pytestmark = pytest.mark.skipif(not uma_available(), reason="UMA not available")


def _conv2d_te_definition(shapes: dict) -> list:
    n, w, h, ci, kw, kh, co = (
        shapes["n"],
        shapes["w"],
        shapes["h"],
        shapes["ci"],
        shapes["kw"],
        shapes["kh"],
        shapes["co"],
    )
    ifmap = te.placeholder((n, ci, w, h), dtype="float32", name="ifmap")
    weights = te.placeholder((co, ci, kw, kh), dtype="float32", name="weights")
    result = topi.nn.conv2d_nchw(ifmap, weights, stride=1, padding=[kw // 2, kh // 2], dilation=1)
    return [ifmap, weights, result]


def _pepare_conv2d_schedule(shapes, use_external_conv2d_impl=True):
    placeholders = _conv2d_te_definition(shapes)

    apps_path = (
        pathlib.Path(str(__file__)).parent.parent.parent.parent.parent.joinpath("apps").absolute()
    )
    conv2d_file = apps_path / "uma" / "_template" / "conv2dnchw.cc"

    with conv2d_file.open() as f:
        sch_tir = _create_schedule(
            placeholders, f, use_external_conv2d_impl=use_external_conv2d_impl
        )
    return placeholders, sch_tir


def _run_external_conv2d(dut_io_arrays, conv2d_shapes, target):
    # Run conv2d with external function
    placeholders, schedule = _pepare_conv2d_schedule(conv2d_shapes)

    uma_lower = UMALower("lower_test")
    uma_lower._tir_passes.append((PassPhase.TIR_PHASE_0, MyAiHwConv2dPass()))
    with tvm.transform.PassContext():
        tir_mod = uma_lower._lower_stir_to_nstir(schedule.mod["main"])

    ifmap_data, weight_data, result_data = dut_io_arrays

    llvm_conv2d_mod = tvm.build(tir_mod, placeholders, target=target, name="test_external_conv2d")
    llvm_conv2d_mod(ifmap_data, weight_data, result_data)


def _run_reference_conv2d(reference_io_arrays, conv2d_shapes, target):
    placeholders, schedule = _pepare_conv2d_schedule(conv2d_shapes)
    ref_mod = tvm.build(schedule.mod, placeholders, target=target, name="test_reference_conv2d")
    ifmap, weights, result = reference_io_arrays
    ref_mod(ifmap, weights, result)


def _prepare_io_arrays(conv2d_shapes, dev):
    dut_io_arrays = _generate_io_arrays(conv2d_shapes, dev)
    _, _, ref_result = _generate_io_arrays(conv2d_shapes, dev)
    reference_io_arrays = [dut_io_arrays[0], dut_io_arrays[1], ref_result]
    return dut_io_arrays, reference_io_arrays


@pytest.mark.parametrize(
    "n, w, h, ci, kw, kh, co",
    [
        (1, 224, 224, 3, 3, 3, 4),
        (1, 224, 224, 3, 5, 5, 4),
        (1, 224, 224, 3, 7, 7, 4),
        (1, 224, 320, 3, 7, 7, 4),
        (1, 224, 224, 3, 7, 7, 4),
    ],
)
def test_lower_with_uma(n, w, h, ci, kw, kh, co):
    target = tvm.target.Target(target="llvm", host="llvm")
    dev = tvm.device(target.kind.name, 0)
    conv2d_shapes = dict(n=n, w=w, h=h, ci=ci, kw=kw, kh=kh, co=co)

    dut_io_arrays, reference_io_arrays = _prepare_io_arrays(conv2d_shapes, dev)

    _run_external_conv2d(dut_io_arrays, conv2d_shapes, target)
    _run_reference_conv2d(reference_io_arrays, conv2d_shapes, target)

    # compare results
    dut_results = dut_io_arrays[2].numpy()
    ref_results = reference_io_arrays[2].numpy()
    tvm.testing.assert_allclose(dut_results, ref_results, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
