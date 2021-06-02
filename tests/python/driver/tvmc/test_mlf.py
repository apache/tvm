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

import tvm
from tvm.driver import tvmc
from tvm.driver.tvmc.main import _main
from tvm.driver.tvmc.model import TVMCPackage, TVMCException


def test_tvmc_cl_compile_run_mlf(tflite_mobilenet_v1_1_quant, tmpdir_factory):
    pytest.importorskip("tflite")

    output_dir = tmpdir_factory.mktemp("mlf")
    input_model = tflite_mobilenet_v1_1_quant
    output_file = os.path.join(output_dir, "mock.tar")

    # Compile the input model and generate a Model Library Format (MLF) archive.
    tvmc_cmd = (
        f"tvmc compile {input_model} --target='llvm' --output {output_file} --output-format mlf"
    )
    tvmc_args = tvmc_cmd.split(" ")[1:]
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


def test_tvmc_import_package_mlf(tflite_compiled_model_mlf):
    pytest.importorskip("tflite")

    # Compile and export a model to a MLF archive so it can be imported.
    exported_tvmc_package = tflite_compiled_model_mlf
    archive_path = exported_tvmc_package.package_path

    # Import the MLF archive. TVMCPackage constructor will call import_package method.
    tvmc_package = TVMCPackage(archive_path)

    assert tvmc_package.lib_name is None, ".lib_name must not be set in the MLF archive."
    assert tvmc_package.lib_path is None, ".lib_path must not be set in the MLF archive."
    assert tvmc_package.graph is not None, ".graph must be set in the MLF archive."
    assert tvmc_package.params is not None, ".params must be set in the MLF archive."
    assert tvmc_package.type == "mlf", ".type must be set to 'mlf' in the MLF format."
