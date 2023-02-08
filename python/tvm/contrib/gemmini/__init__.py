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
Gemmini package is a TVM backend extension to support the Gemmini hardware accelerator
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import tvm._ffi.base

from tvm.relay.backend.contrib.gemmini import *
from .environment import Environment
from .build_module import build_config, lower, build, preprocess_pass
from .helpers import create_header_file
from .utils import *

__version__ = "0.1.0"
