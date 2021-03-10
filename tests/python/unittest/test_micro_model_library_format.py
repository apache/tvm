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

import datetime
import json
import os
import sys
import tarfile

import numpy
import pytest

import tvm
import tvm.relay
from tvm.relay.backend import graph_runtime_factory
import tvm.runtime.module
import tvm.testing
from tvm.contrib import utils


def validate_graph_json(extract_dir, factory):
    with open(os.path.join(extract_dir, "runtime-config", "graph", "graph.json")) as graph_f:
        graph_json = graph_f.read()
        assert graph_json == factory.graph_json

        # Just check it parses and looks roughly right.
        graph = json.loads(graph_json)
        assert "nodes" in graph
        assert len(graph["nodes"]) == 4
        assert "attrs" in graph


@tvm.testing.requires_micro
def test_export_model_library_format_c():
    with utils.TempDirectory.set_keep_for_debug(True):
        target = tvm.target.target.micro("host")
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            relay_mod = tvm.parser.fromtext(
                """
            #[version = "0.0.5"]
            def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), float32], %c : Tensor[(1, 2), float32]) {
            %0 = cast(%a, dtype="float32") + %b * %c;
            %0
            }"""
            )
            factory = tvm.relay.build(
                relay_mod,
                target,
                target_host=target,
                mod_name="add",
                params={"c": numpy.array([[2.0, 4.0]], dtype="float32")},
            )

        temp_dir = utils.tempdir()
        mlf_tar_path = temp_dir.relpath("lib.tar")
        import tvm.micro as micro

        micro.export_model_library_format(factory, mlf_tar_path)
        tf = tarfile.open(mlf_tar_path)

        extract_dir = temp_dir.relpath("extract")
        os.mkdir(extract_dir)
        tf.extractall(extract_dir)

        with open(os.path.join(extract_dir, "metadata.json")) as json_f:
            metadata = json.load(json_f)
            assert metadata["version"] == 1
            assert metadata["model_name"] == "add"
            export_datetime = datetime.datetime.strptime(
                metadata["export_datetime"], "%Y-%m-%d %H:%M:%SZ"
            )
            assert (datetime.datetime.now() - export_datetime) < datetime.timedelta(seconds=60 * 5)
            assert metadata["target"] == {"1": str(target)}
            assert metadata["memory"] == [
                {"storage_id": 0, "size_bytes": 2, "input_binding": "a"},
                {"storage_id": 1, "size_bytes": 8, "input_binding": "b"},
                {"storage_id": 2, "size_bytes": 8, "input_binding": "p0"},
                {"storage_id": 3, "size_bytes": 8},
            ]

        assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "lib0.c"))
        assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "lib1.c"))

        validate_graph_json(extract_dir, factory)

        with open(os.path.join(extract_dir, "relay.txt")) as relay_f:
            assert relay_f.read() == str(relay_mod)

        with open(os.path.join(extract_dir, "parameters", "add.params"), "rb") as params_f:
            params = tvm.relay.load_param_dict(params_f.read())
            assert "p0" in params


@tvm.testing.requires_micro
def test_export_model_library_format_llvm():
    with utils.TempDirectory.set_keep_for_debug(True):
        target = tvm.target.target.micro("host")
        assert str(target)[:2] == "c "
        target = tvm.target.Target("llvm " + str(target)[2:])
        with tvm.transform.PassContext(opt_level=3):
            relay_mod = tvm.parser.fromtext(
                """
            #[version = "0.0.5"]
            def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), float32], %c : Tensor[(1, 2), float32]) {
            %0 = cast(%a, dtype="float32") + %b * %c;
            %0
            }"""
            )
            factory = tvm.relay.build(
                relay_mod,
                target,
                target_host=target,
                mod_name="add",
                params={"c": numpy.array([[2.0, 4.0]], dtype="float32")},
            )

        temp_dir = utils.tempdir()
        mlf_tar_path = temp_dir.relpath("lib.tar")
        import tvm.micro as micro

        micro.export_model_library_format(factory, mlf_tar_path)
        tf = tarfile.open(mlf_tar_path)

        extract_dir = temp_dir.relpath("extract")
        os.mkdir(extract_dir)
        tf.extractall(extract_dir)

        with open(os.path.join(extract_dir, "metadata.json")) as json_f:
            metadata = json.load(json_f)
            assert metadata["version"] == 1
            assert metadata["model_name"] == "add"
            export_datetime = datetime.datetime.strptime(
                metadata["export_datetime"], "%Y-%m-%d %H:%M:%SZ"
            )
            assert (datetime.datetime.now() - export_datetime) < datetime.timedelta(seconds=60 * 5)
            assert metadata["target"] == {"1": str(target)}
            assert metadata["memory"] == [
                {"storage_id": 0, "size_bytes": 2, "input_binding": "a"},
                {"storage_id": 1, "size_bytes": 8, "input_binding": "b"},
                {"storage_id": 2, "size_bytes": 8, "input_binding": "p0"},
                {"storage_id": 3, "size_bytes": 8},
            ]

        assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "lib", "lib0.o"))

        validate_graph_json(extract_dir, factory)

        with open(os.path.join(extract_dir, "relay.txt")) as relay_f:
            assert relay_f.read() == str(relay_mod)

        with open(os.path.join(extract_dir, "parameters", "add.params"), "rb") as params_f:
            params = tvm.relay.load_param_dict(params_f.read())
            assert "p0" in params


@tvm.testing.requires_micro
def test_export_model():
    module = tvm.support.FrontendTestModule()
    factory = graph_runtime_factory.GraphRuntimeFactoryModule(
        None, tvm.target.target.micro("host"), '"graph_json"', module, "test_module", {}
    )

    temp_dir = utils.tempdir()
    import tvm.micro as micro
    import tvm.micro.model_library_format as model_library_format

    with pytest.raises(micro.UnsupportedInModelLibraryFormatError) as exc:
        model_library_format._populate_codegen_dir(module, temp_dir.relpath("codegen"))

        assert str(exc.exception) == (
            "Don't know how to export non-c or non-llvm modules; found: ffi_testing"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
