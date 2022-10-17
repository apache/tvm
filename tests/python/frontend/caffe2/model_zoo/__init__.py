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

"""Store for caffe2 examples and common models."""
from __future__ import absolute_import as _abs
import os
import sys
import importlib
from caffe2.python.models.download import ModelDownloader
from . import squeezenet

models = [
    "squeezenet",
    "resnet50",
    "vgg19",
]

mf = ModelDownloader()


class Model:
    def __init__(self, model_name):
        self.init_net, self.predict_net, self.value_info = mf.get_c2_model(model_name)


for model in models:
    try:
        locals()["c2_" + model] = importlib.import_module("caffe2.python.models." + model)
    except ImportError:
        locals()["c2_" + model] = Model(model)

# squeezenet
def relay_squeezenet():
    return squeezenet.get_workload()
