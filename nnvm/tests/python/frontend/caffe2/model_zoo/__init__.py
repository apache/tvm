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
import importlib

models = [
    'squeezenet',
    'resnet50',
    'vgg19',
]

# skip download if model exist
for model in models:
    try:
        locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
    except ImportError:
        os.system("python -m caffe2.python.models.download -i -f " + model)
        locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
