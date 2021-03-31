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

import numpy as np

from PIL import Image

from tvm.driver import tvmc

from tvm.contrib.download import download_testdata

# Support functions


def download_and_untar(model_url, model_sub_path, temp_dir):
    model_tar_name = os.path.basename(model_url)
    model_path = download_testdata(model_url, model_tar_name, module=["tvmc"])

    if model_path.endswith("tgz") or model_path.endswith("gz"):
        tar = tarfile.open(model_path)
        tar.extractall(path=temp_dir)
        tar.close()

    return os.path.join(temp_dir, model_sub_path)


def get_sample_compiled_module(target_dir):
    """Support function that returns a TFLite compiled module"""
    base_url = "https://storage.googleapis.com/download.tensorflow.org/models"
    model_url = "mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
    model_file = download_and_untar(
        "{}/{}".format(base_url, model_url),
        "mobilenet_v1_1.0_224_quant.tflite",
        temp_dir=target_dir,
    )

    return tvmc.compiler.compile_model(model_file, target="llvm")


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
def onnx_resnet50():
    base_url = "https://github.com/onnx/models/raw/master/vision/classification/resnet/model"
    file_to_download = "resnet50-v2-7.onnx"
    model_file = download_testdata(
        "{}/{}".format(base_url, file_to_download), file_to_download, module=["tvmc"]
    )

    return model_file


@pytest.fixture(scope="session")
def tflite_compiled_module_as_tarfile(tmpdir_factory):

    # Not all CI environments will have TFLite installed
    # so we need to safely skip this fixture that will
    # crash the tests that rely on it.
    # As this is a pytest.fixture, we cannot take advantage
    # of pytest.importorskip. Using the block below instead.
    try:
        import tflite
    except ImportError:
        print("Cannot import tflite, which is required by tflite_compiled_module_as_tarfile.")
        return ""

    target_dir = tmpdir_factory.mktemp("data")
    graph, lib, params, _ = get_sample_compiled_module(target_dir)

    module_file = os.path.join(target_dir, "mock.tar")
    tvmc.compiler.save_module(module_file, graph, lib, params)

    return module_file


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
