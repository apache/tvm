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

pytest.importorskip("tensorflow")

import os
import sys
import tvm
from tensorflow import keras
from tvm.relay.backend.contrib.uma import uma_available
from tvm.driver.tvmc.main import _main

pytestmark = pytest.mark.skipif(not uma_available(), reason="UMA not available")


def run_test(tmpdir_factory, ext_dir_name, check_relay=True):
    if "tvmc_extension" in sys.modules:
        del sys.modules["tvmc_extension"]
    from tvm.driver.tvmc.extensions import _EXTENSIONS

    _EXTENSIONS.clear()
    from tvm.driver.tvmc.composite_target import REGISTERED_CODEGEN

    REGISTERED_CODEGEN.clear()

    tmpdir = tmpdir_factory.mktemp("data")
    model_path = os.path.join(tmpdir, "model.h5")
    package_path = os.path.join(tmpdir, "out.tar")

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=[10, 10, 3], batch_size=1),
            keras.layers.Conv2D(5, kernel_size=(3, 3)),
        ]
    )
    model.save(model_path)

    extension_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ext_dir_name)
    compile_str = (
        f"tvmc compile --target vanilla_accelerator,c -f mlf "
        f"--experimental-tvmc-extension {extension_dir} "
        f"--desired-layout NCHW --dump-code relay "
        f"--output {package_path} {model_path}"
    )
    compile_args = compile_str.split(" ")[1:]
    assert _main(compile_args) == 0
    if check_relay:
        with open(package_path + ".relay") as f:
            assert 'Compiler="vanilla_accelerator"' in f.read()


def test_conv2d(tmpdir_factory):
    run_test(tmpdir_factory, "vanilla_ext")


def test_invalid_ext(tmpdir_factory):
    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError):
            run_test(tmpdir_factory, "invalid_ext", check_relay=False)


if __name__ == "__main__":
    tvm.testing.main()
