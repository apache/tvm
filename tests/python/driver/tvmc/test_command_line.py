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
import logging
import sys

from pytest_lazyfixture import lazy_fixture
from unittest import mock

import tvm
from tvm.driver.tvmc.main import _main
from tvm.driver.tvmc.model import TVMCException
from tvm.driver.tvmc import compiler
from unittest.mock import MagicMock


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


@mock.patch("tvm.relay.build", side_effect=tvm.relay.build)
@mock.patch("tvm.driver.tvmc.model.TVMCPackage.__init__", return_value=None)
def test_tvmc_workspace_pools_check(mock_pkg, mock_relay, keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")
    tmpdir = tmpdir_factory.mktemp("data")

    # Test model compilation
    package_path = os.path.join(tmpdir, "keras-tvm.tar")
    compile_str = (
        f"tvmc compile --target=llvm --workspace-pools=sram "
        f"--workspace-pools-targets=sram:llvm "
        f"--output={package_path} {keras_simple}"
    )
    compile_args = compile_str.split(" ")[1:]
    _main(compile_args)
    assert os.path.exists(package_path)
    assert mock_relay.call_count == 1
    assert mock_relay.call_args_list[0][1]["workspace_memory_pools"].pools[0].pool_name == "sram"


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


def test_tvmc_logger(caplog, tmpdir_factory, keras_simple):
    pytest.importorskip("tensorflow")
    tmpdir = tmpdir_factory.mktemp("out")

    # TUNE
    log_path = os.path.join(tmpdir, "records.json")
    tune_cmd = f"tvmc tune --target llvm -vvvv --output {log_path} " f"--trials 2 {keras_simple}"

    tuning_args = tune_cmd.split(" ")[1:]
    _main(tuning_args)

    # Check that we log during tvmc tune
    for log_str in ("DEBUG", "INFO", "WARNING", "TVMC"):
        assert log_str in caplog.text

    caplog.clear()

    # COMPILE
    module_file = os.path.join(tmpdir, "m.tar")
    compile_cmd = f"tvmc compile --target 'llvm' {keras_simple} -vvvv --output {module_file}"

    compile_args = compile_cmd.split(" ")[1:]
    _main(compile_args)

    # Check that we log during tvmc compile
    for log_str in ("DEBUG", "WARNING", "TVMC"):
        assert log_str in caplog.text

    caplog.clear()

    # RUN
    run_cmd = f"tvmc run -vvvv {module_file}"

    run_args = run_cmd.split(" ")[1:]
    _main(run_args)

    # Check that we log during tvmc run
    for log_str in ("DEBUG", "TVMC"):
        assert log_str in caplog.text


# Unfortunately pytest seems to intercept the logging output, so we can't test whether it
# actually writes the logging output to sys.stdout, but we can test that we call
# logging.basicConfig with the correct arguments
def test_tvmc_logger_set_basicConfig(monkeypatch, tmpdir_factory, keras_simple):
    pytest.importorskip("tensorflow")
    mock_basicConfig = MagicMock()
    monkeypatch.setattr(logging, "basicConfig", mock_basicConfig)

    # Run a random tvmc command
    tmpdir = tmpdir_factory.mktemp("out")
    module_file = os.path.join(tmpdir, "m.tar")
    compile_cmd = f"tvmc compile --target 'llvm' {keras_simple} -vvvv --output {module_file}"
    compile_args = compile_cmd.split(" ")[1:]
    _main(compile_args)

    mock_basicConfig.assert_called_with(stream=sys.stdout)


def test_tvmc_print_pass_times(capsys, keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")
    tmpdir = tmpdir_factory.mktemp("out")
    print_cmd = "--print-pass-times"

    # Compile model
    module_file = os.path.join(tmpdir, "keras-tvm.tar")
    compile_cmd = f"tvmc compile --target 'llvm' {keras_simple} --output {module_file} {print_cmd}"
    compile_args = compile_cmd.split(" ")[1:]
    _main(compile_args)

    # Check for timing results output
    captured_out = capsys.readouterr().out
    for exp_str in ("Compilation time breakdown by pass:", "sequential:", "us]"):
        assert exp_str in captured_out


@pytest.mark.parametrize(
    "print_cmd, out_str",
    [
        (
            "--print-ir-after=[tir.SplitHostDevice]",
            (
                "Print IR after: tir.SplitHostDevice\n# from tvm.script import ir as I\n",
                "@I.ir_module",
            ),
        ),
        (
            "--print-ir-before=[tir.SplitHostDevice]",
            ("Print IR before: tir.SplitHostDevice\n# from tvm.script import ir as I\n"),
        ),
        (
            "--print-ir-after=[tir.ThreadSync,tir.SplitHostDevice]",
            ("tir.ThreadSync,tir.SplitHostDevice"),
        ),
        (
            "--print-ir-before=[tir.SplitHostDevice] --print-ir-after=[tir.SplitHostDevice]",
            ("Print IR before: tir.SplitHostDevice\n", "Print IR after: tir.SplitHostDevice\n"),
        ),
    ],
)
def test_tvmc_print_ir_before_after(capsys, keras_simple, tmpdir_factory, print_cmd, out_str):
    pytest.importorskip("tensorflow")
    tmpdir = tmpdir_factory.mktemp("out")

    # Compile model
    module_file = os.path.join(tmpdir, "keras-tvm.tar")
    compile_cmd = f"tvmc compile --target 'llvm' {keras_simple} --output {module_file} {print_cmd}"
    compile_args = compile_cmd.split(" ")[1:]
    _main(compile_args)

    # Check for printing IR before or IR after
    captured_out = capsys.readouterr().out
    for exp_str in out_str:
        assert exp_str in captured_out
