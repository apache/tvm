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

import logging
import pathlib
from pathlib import Path
from typing import Union
import os
from os import environ
import json

import tvm
import tvm.relay as relay
from tvm.contrib import utils, ndk, graph_executor as runtime
from tvm.contrib.download import download_testdata, download

target = "llvm -mtriple=arm64-linux-android"
target_host = None


def del_dir(target: Union[Path, str], only_if_empty: bool = False):
    target = Path(target).expanduser()
    assert target.is_dir()
    for p in sorted(target.glob("**/*"), reverse=True):
        if not p.exists():
            continue
        p.chmod(0o666)
        if p.is_dir():
            p.rmdir()
        else:
            if only_if_empty:
                raise RuntimeError(f"{p.parent} is not empty!")
            p.unlink()
    target.rmdir()


def get_model(model_name, batch_size=1):
    if model_name == "resnet18_v1":
        import mxnet as mx
        from mxnet import gluon
        from mxnet.gluon.model_zoo import vision

        gluon_model = vision.get_model(model_name, pretrained=True)
        img_size = 224
        data_shape = (batch_size, 3, img_size, img_size)
        net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
        return (net, params)
    elif model_name == "mobilenet_v2":
        import keras
        from keras.applications.mobilenet_v2 import MobileNetV2

        keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
        weights_url = "".join(
            [
                "https://github.com/JonathanCMitchell/",
                "mobilenet_v2_keras/releases/download/v1.1/",
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
            ]
        )
        weights_file = "mobilenet_v2_weights.h5"
        weights_path = download_testdata(weights_url, weights_file, module="keras")
        keras_mobilenet_v2 = MobileNetV2(
            alpha=0.5, include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
        )
        keras_mobilenet_v2.load_weights(weights_path)

        img_size = 224
        data_shape = (batch_size, 3, img_size, img_size)
        mod, params = relay.frontend.from_keras(keras_mobilenet_v2, {"input_1": data_shape})
        return (mod, params)


def main(model_str, output_path):
    if output_path.exists():
        del_dir(output_path)
    output_path.mkdir()
    output_path_str = os.fspath(output_path)
    print(model_str)
    print("getting model...")
    net, params = get_model(model_str)
    try:
        os.mkdir(model_str)
    except FileExistsError:
        pass
    print("building...")
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(net, tvm.target.Target(target, target_host), params=params)
    print("dumping lib...")
    lib.export_library(output_path_str + "/" + "deploy_lib_cpu.so", fcompile=ndk.create_shared)
    print("dumping graph...")
    with open(output_path_str + "/" + "deploy_graph.json", "w") as f:
        f.write(graph)
    print("dumping params...")
    with open(output_path_str + "/" + "deploy_param.params", "wb") as f:
        f.write(tvm.runtime.save_param_dict(params))
    print("dumping labels...")
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_path = output_path_str + "/image_net_labels"
    download(synset_url, output_path_str + "/image_net_labels")
    with open(synset_path) as fi:
        synset = eval(fi.read())
        with open(output_path_str + "/image_net_labels.json", "w") as fo:
            json.dump(synset, fo, indent=4)
    os.remove(synset_path)


if __name__ == "__main__":
    if environ.get("TVM_NDK_CC") is None:
        raise RuntimeError("Require environment variable TVM_NDK_CC")
    models_path = Path().absolute().parent.joinpath("app/src/main/assets/models/")
    if not models_path.exists():
        models_path.mkdir(parents=True)
    models = {
        "mobilenet_v2": models_path.joinpath("mobilenet_v2"),
        "resnet18_v1": models_path.joinpath("resnet18_v1"),
    }
    for model, output_path in models.items():
        main(model, output_path)
