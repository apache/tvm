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
import shlex
import sys

import tvm
import tvm.testing
from tvm.autotvm.measure.executor import Executor
from tvm.driver import tvmc
from tvm.driver.tvmc.main import _main
from tvm.driver.tvmc.model import TVMCPackage, TVMCException
from tvm.relay import backend


def test_tvmc_cl_compile_run_mlf(tflite_mobilenet_v1_1_quant, tmpdir_factory):
    target = "c"
    executor = "aot"
    pass_configs = ["tir.disable_vectorize=1"]
    pytest.importorskip("tflite")

    output_dir = tmpdir_factory.mktemp("mlf")
    input_model = tflite_mobilenet_v1_1_quant
    output_file = os.path.join(output_dir, "mock.tar")

    # Compile the input model and generate a Model Library Format (MLF) archive.
    pass_config_args = " ".join([f"--pass-config {pass_config}" for pass_config in pass_configs])
    tvmc_cmd = f"tvmc compile {input_model} --target={target} --executor={executor} {pass_config_args} --output {output_file} --output-format mlf"
    tvmc_args = shlex.split(tvmc_cmd)[1:]
    _main(tvmc_args)
    assert os.path.exists(output_file), "Could not find the exported MLF archive."

    # Run the MLF archive. It must fail since it's only supported on micro targets.
    tvmc_cmd = f"tvmc run {output_file}"
    tvmc_args = tvmc_cmd.split(" ")[1:]
    exit_code = _main(tvmc_args)
    on_error = "Trying to run a MLF archive must fail because it's only supported on micro targets."
    assert exit_code != 0, on_error


def test_tvmc_export_package_mlf(tflite_mobilenet_v1_1_quant, tmpdir_factory):
    pytest.importorskip("tflite")

    tvmc_model = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant)
    mod, params = tvmc_model.mod, tvmc_model.params

    graph_module = tvm.relay.build(mod, target="llvm", params=params)

    output_dir = tmpdir_factory.mktemp("mlf")
    output_file = os.path.join(output_dir, "mock.tar")

    # Try to export MLF with no cross compiler set. No exception must be thrown.
    tvmc_model.export_package(
        executor_factory=graph_module,
        package_path=output_file,
        cross=None,
        output_format="mlf",
    )
    assert os.path.exists(output_file), "Could not find the exported MLF archive."

    # Try to export a MLF whilst also specifying a cross compiler. Since
    # that's not supported it must throw a TVMCException and report the
    # reason accordingly.
    with pytest.raises(TVMCException) as exp:
        tvmc_model.export_package(
            executor_factory=graph_module,
            package_path=output_file,
            cross="cc",
            output_format="mlf",
        )
    expected_reason = "Specifying the MLF output and a cross compiler is not supported."
    on_error = "A TVMCException was caught but its reason is not the expected one."
    assert str(exp.value) == expected_reason, on_error


def test_tvmc_import_package_project_dir(tflite_mobilenet_v1_1_quant, tflite_compile_model):
    pytest.importorskip("tflite")

    # Generate a MLF archive.
    compiled_model_mlf_tvmc_package = tflite_compile_model(
        tflite_mobilenet_v1_1_quant, output_format="mlf"
    )

    # Import the MLF archive setting 'project_dir'. It must succeed.
    mlf_archive_path = compiled_model_mlf_tvmc_package.package_path
    tvmc_package = TVMCPackage(mlf_archive_path, project_dir="/tmp/foobar")
    assert tvmc_package.type == "mlf", "Can't load the MLF archive passing the project directory!"

    # Generate a Classic archive.
    compiled_model_classic_tvmc_package = tflite_compile_model(tflite_mobilenet_v1_1_quant)

    # Import the Classic archive setting 'project_dir'.
    # It must fail since setting 'project_dir' is only support when importing a MLF archive.
    classic_archive_path = compiled_model_classic_tvmc_package.package_path
    with pytest.raises(TVMCException) as exp:
        tvmc_package = TVMCPackage(classic_archive_path, project_dir="/tmp/foobar")

    expected_reason = "Setting 'project_dir' is only allowed when importing a MLF.!"
    on_error = "A TVMCException was caught but its reason is not the expected one."
    assert str(exp.value) == expected_reason, on_error


def test_tvmc_import_package_mlf_graph(tflite_mobilenet_v1_1_quant, tflite_compile_model):
    pytest.importorskip("tflite")

    tflite_compiled_model_mlf = tflite_compile_model(
        tflite_mobilenet_v1_1_quant, output_format="mlf"
    )

    # Compile and export a model to a MLF archive so it can be imported.
    exported_tvmc_package = tflite_compiled_model_mlf
    archive_path = exported_tvmc_package.package_path

    # Import the MLF archive. TVMCPackage constructor will call import_package method.
    tvmc_package = TVMCPackage(archive_path)

    assert tvmc_package.lib_name is None, ".lib_name must not be set in the MLF archive."
    assert tvmc_package.lib_path is None, ".lib_path must not be set in the MLF archive."
    assert (
        tvmc_package.graph is not None
    ), ".graph must be set in the MLF archive for Graph executor."
    assert tvmc_package.params is not None, ".params must be set in the MLF archive."
    assert tvmc_package.type == "mlf", ".type must be set to 'mlf' in the MLF format."


def test_tvmc_import_package_mlf_aot(tflite_mobilenet_v1_1_quant, tflite_compile_model):
    pytest.importorskip("tflite")

    tflite_compiled_model_mlf = tflite_compile_model(
        tflite_mobilenet_v1_1_quant,
        target="c",
        executor=backend.Executor("aot"),
        output_format="mlf",
        pass_context_configs=["tir.disable_vectorize=1"],
    )

    # Compile and export a model to a MLF archive so it can be imported.
    exported_tvmc_package = tflite_compiled_model_mlf
    archive_path = exported_tvmc_package.package_path

    # Import the MLF archive. TVMCPackage constructor will call import_package method.
    tvmc_package = TVMCPackage(archive_path)

    assert tvmc_package.lib_name is None, ".lib_name must not be set in the MLF archive."
    assert tvmc_package.lib_path is None, ".lib_path must not be set in the MLF archive."
    assert tvmc_package.graph is None, ".graph must not be set in the MLF archive for AOT executor."
    assert tvmc_package.params is not None, ".params must be set in the MLF archive."
    assert tvmc_package.type == "mlf", ".type must be set to 'mlf' in the MLF format."


if __name__ == "__main__":
    tvm.testing.main()
