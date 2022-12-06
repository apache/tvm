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
"""
Utils to help generate the MobileNet TFLite model
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import os
from tvm.contrib.download import download_testdata
import numpy as np
import tensorflow as tf


def get_real_image(im_height, im_width):
    from PIL import Image

    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


def run_tflite_model(tflite_model_buf, input_data):
    """Generic function to execute TFLite"""
    try:
        from tensorflow import lite as interpreter_wrapper
    except ImportError:
        from tensorflow.contrib import lite as interpreter_wrapper

    input_data = input_data if isinstance(input_data, list) else [input_data]

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]["index"], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]["index"]))

    return tflite_output


def download_model():
    model_url = (
        "https://storage.googleapis.com/download.tensorflow.org/models/"
        "tflite_11_05_08/mobilenet_v2_1.0_224.tgz"
    )

    # Download model tar file and extract it to get mobilenet_v2_1.0_224.tflite
    model_path = download_testdata(
        model_url, "mobilenet_v2_1.0_224.tgz", module=["tf", "official", "mobilenet_v2"]
    )
    model_dir = os.path.dirname(model_path)

    return model_dir, model_path


def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


def create_tflite_model(model_dir: str):
    # tflite_model_name = [f for f in os.listdir(model_dir) if f.endswith(".tflite")][0]
    # return f"{model_dir}/{tflite_model_name}"
    def representative_data_gen():
        dataset = [
            np.array(np.random.randint(0, 255, size=(1, 224, 224, 3)), dtype=np.float32)
            for s in range(100)
        ]
        for input_value in dataset:
            # Model has only one input so each data point has one element.s
            yield [input_value]

    pb_file = [f for f in os.listdir(model_dir) if f.endswith(".pb")][0]
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        f"{model_dir}/{pb_file}",
        input_arrays=["input"],
        input_shapes={"input": [1, 224, 224, 3]},
        output_arrays=["MobilenetV2/Predictions/Reshape"],
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_data_gen
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()
    tflite_model_name = pb_file.replace(".pb", ".tflite")
    with open(f"{model_dir}/{tflite_model_name}", "wb") as f:
        f.write(tflite_model)

    return f"{model_dir}/{tflite_model_name}"


def generate_mobilenet_tflite_model():
    model_dir, model_path = download_model()
    extract(model_path)
    return create_tflite_model(model_dir)
