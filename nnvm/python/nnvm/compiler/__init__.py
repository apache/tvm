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

"""NNVM compiler toolchain.

User only need to use :any:`build` and :any:`build_config` to do the compilation,
and :any:`save_param_dict` to save the parameters into bytes.
The other APIs are for more advanced interaction with the compiler toolchain.
"""
from __future__ import absolute_import

import tvm

from . import build_module
from . build_module import build, optimize, build_config
from . compile_engine import engine, graph_key
from . param_dict import save_param_dict, load_param_dict

from .. import symbol as _symbol
from .. import graph as _graph

from .. import top as _top


tvm.register_extension(_symbol.Symbol, _symbol.Symbol)
tvm.register_extension(_graph.Graph, _graph.Graph)
