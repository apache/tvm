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
Frontends for constructing Relay programs.

Contains the model importers currently defined
for Relay.
"""

from __future__ import absolute_import

from .mxnet import from_mxnet
from .mxnet_qnn_op_utils import quantize_conv_bias_mkldnn_from_var
from .keras import from_keras
from .onnx import from_onnx
from .tflite import from_tflite
from .coreml import from_coreml
from .caffe2 import from_caffe2
from .tensorflow import from_tensorflow
from .darknet import from_darknet
from .pytorch import from_pytorch
