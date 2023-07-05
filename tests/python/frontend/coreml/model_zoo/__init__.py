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
"""coreml model zoo for testing purposes."""
import os
from PIL import Image
import numpy as np
from tvm.contrib.download import download_testdata


def get_mobilenet():
    url = "https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel"
    dst = "mobilenet.mlmodel"
    real_dst = download_testdata(url, dst, module="coreml")
    return os.path.abspath(real_dst)


def get_resnet50():
    url = "https://docs-assets.developer.apple.com/coreml/models/Resnet50.mlmodel"
    dst = "resnet50.mlmodel"
    real_dst = download_testdata(url, dst, module="coreml")
    return os.path.abspath(real_dst)


def get_cat_image():
    """Get cat image"""
    url = (
        "https://gist.githubusercontent.com/zhreshold/"
        + "bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png"
    )
    dst = "cat.png"
    real_dst = download_testdata(url, dst, module="data")
    img = Image.open(real_dst).resize((224, 224))
    # CoreML's standard model image format is BGR
    img_bgr = np.array(img)[:, :, ::-1]
    img = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]
    return np.asarray(img)
