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
from utils.adreno_utils import build_run_compare, get_model, gpu_preprocess


def convert_to_fp16(mod, dtype):
    from tvm.ir import IRModule

    mod = IRModule.from_expr(mod)
    seq = tvm.transform.Sequential(
        [relay.transform.InferType(), relay.transform.ToMixedPrecision()]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
        return mod


def _test_mobilenet_v1(remote, target, dtype):
    mod, params, inputs, dtypes = get_model(
        "https://github.com/mlcommons/mobile_models/raw/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_float.tflite",
        "mobilenet_edgetpu_224_1.0_float.tflite",
        "tflite",
    )
    if dtype == "float16":
        mod = convert_to_fp16(mod["main"], dtype)
    build_run_compare(remote, mod, params, inputs, dtypes, target, [])


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/13443")
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
@pytest.mark.skipif(tvm.testing.utils.IS_IN_CI, reason="CI doesn't support fp16(half datatypes)")
def test_mobilenet_v1_fp16(remote, target):
    _test_mobilenet_v1(remote, target, "float16")


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/13443")
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mobilenet_v1_fp32(remote, target):
    _test_mobilenet_v1(remote, target, "float32")


if __name__ == "__main__":
    tvm.testing.main()
