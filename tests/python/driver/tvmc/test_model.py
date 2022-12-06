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
import platform
import pytest
import os
import numpy as np

from os import path

from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel, TVMCPackage, TVMCResult
from tvm.runtime.module import BenchmarkResult


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
@pytest.mark.parametrize("use_vm", [True, False])
def test_tvmc_workflow(use_vm, keras_simple):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    # Reset so the input name remains consistent across unit test runs
    tf.keras.backend.clear_session()

    tvmc_model = tvmc.load(keras_simple)
    tuning_records = tvmc.tune(tvmc_model, target="llvm", enable_autoscheduler=True, trials=2)
    tvmc_package = tvmc.compile(
        tvmc_model, tuning_records=tuning_records, target="llvm", use_vm=use_vm
    )
    input_dict = {"input_1": np.random.uniform(size=(1, 32, 32, 3)).astype("float32")}

    result = tvmc.run(
        tvmc_package, device="cpu", end_to_end=True, benchmark=True, inputs=input_dict
    )
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_package) is TVMCPackage
    assert type(result) is TVMCResult
    assert path.exists(tuning_records)
    assert type(result.outputs) is dict
    assert type(result.times) is BenchmarkResult
    assert "output_0" in result.outputs.keys()


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
@pytest.mark.parametrize("use_vm", [True, False])
def test_save_load_model(use_vm, keras_simple, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir = tmpdir_factory.mktemp("data")
    tvmc_model = tvmc.load(keras_simple)

    # Create tuning artifacts
    tvmc.tune(tvmc_model, target="llvm", trials=2)

    # Create package artifacts
    tvmc.compile(tvmc_model, target="llvm", use_vm=use_vm)

    # Save the model to disk
    model_path = os.path.join(tmpdir, "saved_model.tar")
    tvmc_model.save(model_path)

    # Load the model into a new TVMCModel
    new_tvmc_model = TVMCModel(model_path=model_path)

    # Check that the two models match.
    assert str(new_tvmc_model.mod) == str(tvmc_model.mod)
    # Check that tuning records and the compiled package are recoverable.
    assert path.exists(new_tvmc_model.default_package_path())
    assert path.exists(new_tvmc_model.default_tuning_records_path())
