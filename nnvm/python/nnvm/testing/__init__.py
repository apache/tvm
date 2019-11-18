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

"""Utilities for testing and benchmarks"""
from __future__ import absolute_import as _abs

from .config import ctx_list
from .utils import create_workload
from . import mobilenet
from . import mobilenet_v2
from . import mlp
from . import resnet
from . import vgg
from . import densenet
from . import squeezenet
from . import inception_v3
from . import dcgan
from . import dqn
from . import check_computation
