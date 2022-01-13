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
import os

from tvm.driver.tvmc.main import _main


def test_tvmc_cl_workflow(keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")

    tmpdir = tmpdir_factory.mktemp("data")

    # Test model tuning
    log_path = os.path.join(tmpdir, "keras-autotuner_records.json")
    tuning_str = (
        f"tvmc tune --target llvm --output {log_path} "
        f"--trials 2 --enable-autoscheduler {keras_simple}"
    )
    tuning_args = tuning_str.split(" ")[1:]
    _main(tuning_args)
    assert os.path.exists(log_path)

    # Test model compilation
    package_path = os.path.join(tmpdir, "keras-tvm.tar")
    compile_str = (
        f"tvmc compile --target llvm --tuning-records {log_path} "
        f"--output {package_path} {keras_simple}"
    )
    compile_args = compile_str.split(" ")[1:]
    _main(compile_args)
    assert os.path.exists(package_path)

    # Test running the model
    output_path = os.path.join(tmpdir, "predictions.npz")
    run_str = f"tvmc run --outputs {output_path} {package_path}"
    run_args = run_str.split(" ")[1:]
    _main(run_args)
    assert os.path.exists(output_path)
