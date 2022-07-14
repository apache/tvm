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
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER

from tvm.testing.aot import compile_and_run, AOTTestModel, AOTTestRunner

import tvm
from tvm import relay
from tvm.relay.backend.contrib.uma._template.backend import MyAiHwBackend
from tvm.relay import transform
from collections import OrderedDict

import numpy as np
import tarfile
from pathlib import Path
import onnx

from tvm.testing.aot import (
    AOTTestModel,
    AOTTestRunner,
    generate_ref_data,
    compile_and_run,
)


def create_conv2d(groups=1, test_runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    test_runner = AOTTestRunner(
        makefile=test_runner.makefile,
        prologue=test_runner.prologue,
        epilogue=test_runner.epilogue,
        includes=test_runner.includes,
        parameters=test_runner.parameters,
        pass_config=pass_config,
    )
    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, test_runner


def main():
    mod, inputs, output_list, test_runner = create_conv2d()

    uma_backend = MyAiHwBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("my_ai_hw", host=tvm.target.Target("c"))

    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    print(f"Generated files are in {export_directory}")
    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        test_runner,
        interface_api="c",
        use_unpacked_api=True,
        target=target,
        test_dir=str(export_directory),
    )


if __name__ == "__main__":
    main()
