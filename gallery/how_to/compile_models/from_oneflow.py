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
Compile OneFlow Models
======================
**Author**: `Xiaoyu Zhang <https://github.com/BBuf/>`_

This article is an introductory tutorial to deploy OneFlow models with Relay.

For us to begin with, OneFlow package should be installed.

A quick solution is to install via pip

.. code-block:: bash

    pip install flowvision==0.1.0
    python3 -m pip install -f https://release.oneflow.info oneflow==0.7.0+cpu

or please refer to official site:
https://github.com/Oneflow-Inc/oneflow

Currently, TVM supports OneFlow 0.7.0. Other versions may be unstable.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
import os, math
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# oneflow imports
import flowvision
import oneflow as flow
import oneflow.nn as nn

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
