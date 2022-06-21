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
import platform
import pytest
import shutil

from pytest_lazyfixture import lazy_fixture
from unittest import mock
from tvm.driver.tvmc.main import _main
from tvm.driver.tvmc.model import TVMCException
from tvm.driver.tvmc import compiler


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
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
    run_str = f"tvmc run --end-to-end --outputs {output_path} {package_path}"
    run_args = run_str.split(" ")[1:]
    _main(run_args)
    assert os.path.exists(output_path)


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_tvmc_cl_workflow_json_config(keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")
    tune_config_file = "tune_config_test"
    tmpdir = tmpdir_factory.mktemp("data")

    # Test model tuning
    log_path = os.path.join(tmpdir, "keras-autotuner_records.json")
    tuning_str = (
        f"tvmc tune --config {tune_config_file} --output {log_path} "
        f"--enable-autoscheduler {keras_simple}"
    )
    tuning_args = tuning_str.split(" ")[1:]
    _main(tuning_args)
    assert os.path.exists(log_path)

    # Test model compilation
    package_path = os.path.join(tmpdir, "keras-tvm.tar")
    compile_str = (
        f"tvmc compile --tuning-records {log_path} " f"--output {package_path} {keras_simple}"
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


@pytest.fixture
def missing_file():
    missing_file_name = "missing_file_as_invalid_input.tfite"
    return missing_file_name


@pytest.fixture
def broken_symlink(tmp_path):
    broken_symlink = "broken_symlink_as_invalid_input.tflite"
    os.symlink("non_existing_file", tmp_path / broken_symlink)
    yield broken_symlink
    os.unlink(tmp_path / broken_symlink)


@pytest.fixture
def fake_directory(tmp_path):
    dir_as_invalid = "dir_as_invalid_input.tflite"
    os.mkdir(tmp_path / dir_as_invalid)
    yield dir_as_invalid
    shutil.rmtree(tmp_path / dir_as_invalid)


@pytest.mark.parametrize(
    "invalid_input",
    [lazy_fixture("missing_file"), lazy_fixture("broken_symlink"), lazy_fixture("fake_directory")],
)
def test_tvmc_compile_file_check(capsys, invalid_input):
    compile_cmd = f"tvmc compile --target 'c' {invalid_input}"
    run_arg = compile_cmd.split(" ")[1:]

    _main(run_arg)

    captured = capsys.readouterr()
    expected_err = (
        f"Error: Input file '{invalid_input}' doesn't exist, "
        "is a broken symbolic link, or a directory.\n"
    )
    on_assert_error = f"'tvmc compile' failed to check invalid FILE: {invalid_input}"
    assert captured.err == expected_err, on_assert_error


@pytest.mark.parametrize(
    "invalid_input",
    [lazy_fixture("missing_file"), lazy_fixture("broken_symlink"), lazy_fixture("fake_directory")],
)
def test_tvmc_tune_file_check(capsys, invalid_input):
    tune_cmd = f"tvmc tune --target 'llvm' --output output.json {invalid_input}"
    run_arg = tune_cmd.split(" ")[1:]

    _main(run_arg)

    captured = capsys.readouterr()
    expected_err = (
        f"Error: Input file '{invalid_input}' doesn't exist, "
        "is a broken symbolic link, or a directory.\n"
    )
    on_assert_error = f"'tvmc tune' failed to check invalid FILE: {invalid_input}"
    assert captured.err == expected_err, on_assert_error


@pytest.fixture
def paddle_model(paddle_resnet50):
    # If we can't import "paddle" module, skip testing paddle as the input model.
    if pytest.importorskip("paddle", reason="'paddle' module not installed"):
        return paddle_resnet50


@pytest.mark.parametrize(
    "model",
    [
        lazy_fixture("paddle_model"),
    ],
)
# compile_model() can take too long and is tested elsewhere, hence it's mocked below
@mock.patch.object(compiler, "compile_model")
# @mock.patch.object(compiler, "compile_model")
def test_tvmc_compile_input_model(mock_compile_model, tmpdir_factory, model):

    output_dir = tmpdir_factory.mktemp("output")
    output_file = output_dir / "model.tar"

    compile_cmd = (
        f"tvmc compile --target 'llvm' {model} --model-format paddle --output {output_file}"
    )
    run_arg = compile_cmd.split(" ")[1:]

    _main(run_arg)

    mock_compile_model.assert_called_once()
