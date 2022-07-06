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

import tvm
from tvm import relay, IRModule
from tvm.contrib.download import download_testdata
from tvm.relay.backend.contrib.uma._template.backend import MyAiHwBackend

import numpy as np
import tarfile
from pathlib import Path

import onnx


def import_mnist12() -> [IRModule, dict]:
    model_url = "".join(
        ["https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx"])
    model_path = download_testdata(model_url, "mnist-12.onnx", module="onnx")
    onnx_model = onnx.load(model_path)
    input_name = "Input3"
    shape_dict = {input_name: (1, 1, 28, 28)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params


def import_restnet50() -> [IRModule, dict]:
    model_url = "".join(
        ["https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"])
    model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
    onnx_model = onnx.load(model_path)
    input_name = "data"
    shape_dict = {input_name: (1, 3, 224, 224)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params


def main():

    mod, params = import_mnist12()

    print(mod)

    # Relay target specific partitioning
    uma_backend = MyAiHwBackend()
    uma_backend.register()

    @tvm.register_func("relay.ext.my_ai_hw")
    def uma_compiler(ref):
        print(ref)

    mod = uma_backend.partition(mod)

    # Relay build (AOT C target)
    TARGET = tvm.target.Target("my_ai_hw", host=tvm.target.Target("c"))
    GENERIC_TARGET = tvm.target.Target("c")
    RUNTIME = tvm.relay.backend.Runtime("crt")
    EXECUTOR = tvm.relay.backend.Executor(
        "aot",
        {
            "workspace-byte-alignment": 8,
            #"unpacked-api": True
        },
    )

    with tvm.transform.PassContext(
        opt_level=3,
        config={"tir.disable_vectorize": True,
                #"tir.disable_storage_rewrite": True,
                "tir.usmp.enable": True,
                "tir.usmp.algorithm": "greedy_by_conflicts"
                },
        disabled_pass=["AlterOpLayout"]
    ):
        module = relay.build(mod, target=[GENERIC_TARGET, TARGET], runtime=RUNTIME, executor=EXECUTOR, params=params)

    model_library_format_tar_path = Path("build/lib.tar")
    model_library_format_tar_path.unlink(missing_ok=True)
    model_library_format_tar_path.parent.mkdir(parents=True, exist_ok=True)

    tvm.micro.export_model_library_format(module, model_library_format_tar_path)

    print("Built MLF Library: ")
    with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
        print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))
        tar_f.extractall(model_library_format_tar_path.parent)





if __name__ == "__main__":
    main()
