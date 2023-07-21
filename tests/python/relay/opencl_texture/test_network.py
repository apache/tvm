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

import re

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.contrib import utils
from tvm.relay import testing
from tvm.relay.op import register_mixed_precision_conversion
from utils.adreno_utils import build_run_compare, build_run_compare_vm, get_model, gpu_preprocess


executor_type = tvm.testing.parameter("ge", "vm")


def _test_mobilenet_v1(remote, target, calc_dtype, executor_type, acc_dtype):
    mod, params, inputs, dtypes = get_model(
        "https://github.com/mlcommons/mobile_models/raw/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_float.tflite",
        "mobilenet_edgetpu_224_1.0_float.tflite",
        "tflite",
    )
    if calc_dtype == "float16":
        from tvm.driver.tvmc.transform import apply_graph_transforms

        mod = apply_graph_transforms(
            mod,
            {
                "mixed_precision": True,
                "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
                "mixed_precision_calculation_type": calc_dtype,
                "mixed_precision_acc_type": acc_dtype,
            },
        )

    if executor_type == "ge":
        build_run_compare(remote, mod, params, inputs, dtypes, target, [])
    else:
        build_run_compare_vm(remote, mod, params, inputs, dtypes, target, [])


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/13443")
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
@pytest.mark.skipif(tvm.testing.utils.IS_IN_CI, reason="CI doesn't support fp16(half datatypes)")
def test_mobilenet_v1_fp16(remote, target, executor_type):
    _test_mobilenet_v1(remote, target, "float16", executor_type, "float16")


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/13443")
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mobilenet_v1_fp32(remote, target, executor_type):
    _test_mobilenet_v1(remote, target, "float32", executor_type, "float32")


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/13443")
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mobilenet_v1_fp16_acc32(remote, target, executor_type):
    _test_mobilenet_v1(remote, target, "float16", executor_type, "float32")


if __name__ == "__main__":
    tvm.testing.main()
