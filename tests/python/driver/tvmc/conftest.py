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
import pytest
import tarfile
import textwrap

import numpy as np

from PIL import Image

import tvm
from tvm import relay
from tvm.driver import tvmc

from tvm.contrib.download import download_testdata

# Support functions


def download_and_untar(model_url, model_sub_path, temp_dir):
    model_tar_name = os.path.basename(model_url)
    model_path = download_testdata(model_url, model_tar_name, module=["tvmc"])

    if model_path.endswith("tgz") or model_path.endswith("gz") or model_path.endswith("tar"):
        tar = tarfile.open(model_path)
        tar.extractall(path=temp_dir)
        tar.close()

    return os.path.join(temp_dir, model_sub_path)


# PyTest fixtures


@pytest.fixture(scope="session")
def tflite_mobilenet_v1_1_quant(tmpdir_factory):
    base_url = "https://storage.googleapis.com/download.tensorflow.org/models"
    model_url = "mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
    model_file = download_and_untar(
        "{}/{}".format(base_url, model_url),
        "mobilenet_v1_1.0_224_quant.tflite",
        temp_dir=tmpdir_factory.mktemp("data"),
    )

    return model_file


@pytest.fixture(scope="session")
def pb_mobilenet_v1_1_quant(tmpdir_factory):
    base_url = "https://storage.googleapis.com/download.tensorflow.org/models"
    model_url = "mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"
    model_file = download_and_untar(
        "{}/{}".format(base_url, model_url),
        "mobilenet_v1_1.0_224_frozen.pb",
        temp_dir=tmpdir_factory.mktemp("data"),
    )

    return model_file


@pytest.fixture(scope="session")
def keras_resnet50(tmpdir_factory):
    try:
        from tensorflow.keras.applications.resnet50 import ResNet50
    except ImportError:
        # not all environments provide TensorFlow, so skip this fixture
        # if that is that case.
        return ""

    model_file_name = "{}/{}".format(tmpdir_factory.mktemp("data"), "resnet50.h5")
    model = ResNet50(include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000)
    model.save(model_file_name)

    return model_file_name


@pytest.fixture(scope="session")
def keras_simple(tmpdir_factory):
    try:
        from tensorflow import keras
    except ImportError:
        # not all environments provide TensorFlow, so skip this fixture
        # if that is that case.
        return ""

    model_file_name = "{}/{}".format(tmpdir_factory.mktemp("data"), "simple_conv.h5")
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=[32, 32, 3], batch_size=1),
            keras.layers.Conv2D(8, kernel_size=(3, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(64),
        ]
    )
    model.save(model_file_name)

    return model_file_name


@pytest.fixture(scope="session")
def pytorch_resnet18(tmpdir_factory):
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        # Not all environments provide Pytorch, so skip if that's the case.
        return ""
    model = models.resnet18()
    model_file_name = "{}/{}".format(tmpdir_factory.mktemp("data"), "resnet18.pth")
    # Trace model into torchscript.
    traced_cpu = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    torch.jit.save(traced_cpu, model_file_name)

    return model_file_name


@pytest.fixture(scope="session")
def pytorch_mobilenetv2_quantized(tmpdir_factory):
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        # Not all environments provide Pytorch, so skip if that's the case.
        return ""
    model = models.quantization.mobilenet_v2(quantize=True)
    model_file_name = "{}/{}".format(tmpdir_factory.mktemp("data"), "mobilenet_v2_quantized.pth")
    # Trace model into torchscript.
    traced_cpu = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    torch.jit.save(traced_cpu, model_file_name)

    return model_file_name


@pytest.fixture(scope="session")
def onnx_resnet50():
    base_url = "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/resnet/model"
    file_to_download = "resnet50-v2-7.onnx"
    model_file = download_testdata(
        "{}/{}".format(base_url, file_to_download), file_to_download, module=["tvmc"]
    )

    return model_file


@pytest.fixture(scope="session")
def paddle_resnet50(tmpdir_factory):
    base_url = "https://bj.bcebos.com/x2paddle/models"
    model_url = "paddle_resnet50.tar"
    model_file = download_and_untar(
        "{}/{}".format(base_url, model_url),
        "paddle_resnet50/model.pdmodel",
        temp_dir=tmpdir_factory.mktemp("data"),
    )
    return model_file


@pytest.fixture(scope="session")
def onnx_mnist():
    base_url = "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/mnist/model"
    file_to_download = "mnist-1.onnx"
    model_file = download_testdata(
        "{}/{}".format(base_url, file_to_download), file_to_download, module=["tvmc"]
    )

    return model_file


@pytest.fixture
def tflite_compile_model(tmpdir_factory):
    """Support function that returns a TFLite compiled module"""

    def model_compiler(model_file, **overrides):
        package_path = tmpdir_factory.mktemp("data").join("mock.tar")
        tvmc_model = tvmc.frontends.load_model(model_file)
        args = {"target": "llvm", **overrides}
        return tvmc.compiler.compile_model(tvmc_model, package_path=package_path, **args)

    # Returns a TVMCPackage
    return model_compiler


@pytest.fixture
def relay_compile_model(tmpdir_factory):
    """Support function that returns a TFLite compiled module"""

    def model_compiler(model_file, shape_dict, **overrides):
        package_path = tmpdir_factory.mktemp("data").join("mock.tar")
        tvmc_model = tvmc.frontends.load_model(
            model_file, model_format="relay", shape_dict=shape_dict
        )
        args = {"target": "llvm", **overrides}
        return tvmc.compiler.compile_model(tvmc_model, package_path=package_path, **args)

    # Returns a TVMCPackage
    return model_compiler


@pytest.fixture(scope="session")
def imagenet_cat(tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    cat_file_name = "imagenet_cat.npz"

    cat_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    image_path = download_testdata(cat_url, "inputs", module=["tvmc"])
    resized_image = Image.open(image_path).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")
    image_data = np.expand_dims(image_data, axis=0)

    cat_file_full_path = os.path.join(tmpdir_name, cat_file_name)
    np.savez(cat_file_full_path, input=image_data)

    return cat_file_full_path


@pytest.fixture(scope="session")
def tflite_mobilenet_v1_0_25_128(tmpdir_factory):
    base_url = "https://storage.googleapis.com/download.tensorflow.org/models"
    model_url = "mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz"
    model_file = download_and_untar(
        "{}/{}".format(base_url, model_url),
        "mobilenet_v1_0.25_128.tflite",
        temp_dir=tmpdir_factory.mktemp("data"),
    )

    return model_file


@pytest.fixture(scope="session")
def tflite_cnn_s_quantized(tmpdir_factory):
    base_url = "https://github.com/ARM-software/ML-zoo/raw/48a22ee22325d15d2371a6df24eb7d67e21dcc97/models/keyword_spotting/cnn_small/tflite_int8"
    file_to_download = "cnn_s_quantized.tflite"
    model_file = download_testdata(
        "{}/{}".format(base_url, file_to_download), file_to_download, module=["tvmc"]
    )
    return model_file


@pytest.fixture(scope="session")
def relay_text_conv2d(tmpdir_factory):
    file_path = os.path.join(tmpdir_factory.mktemp("model"), "relay.txt")

    RELAY_MODEL = textwrap.dedent(
        """\
        #[version = "0.0.5"]
        def @main(%data : Tensor[(1, 3, 64, 64), uint8], %weight : Tensor[(3, 3, 5, 5), int8]) {
            %1 = nn.conv2d(
                 %data,
                 %weight,
                 padding=[2, 2],
                 channels=3,
                 kernel_size=[5, 5],
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 out_dtype="int32");
            %2 = cast(nn.max_pool2d(%1, pool_size=[3, 3]), dtype="int8");
            %3 = nn.conv2d(
                 %2,
                 %weight,
                 padding=[2, 2],
                 channels=3,
                 kernel_size=[5, 5],
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 out_dtype="int32");
            %4 = nn.max_pool2d(%3, pool_size=[3, 3]);
            %4
        }
    """
    )

    with open(file_path, "w") as relay_text:
        relay_text.write(RELAY_MODEL)
    return file_path


@pytest.fixture(scope="session")
def relay_conv2d():
    """
    Simple conv2d Relay implementation.
    """
    dtype = "float32"

    x = relay.var("x", shape=(1, 4, 2, 2), dtype=dtype)
    weight = relay.const(np.random.uniform(size=(2, 4, 2, 2)), dtype=dtype)
    x = relay.nn.conv2d(x, weight)
    func = relay.Function(relay.analysis.free_vars(x), x)
    return tvm.IRModule.from_expr(func)
