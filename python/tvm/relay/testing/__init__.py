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

import tvm.relay as relay
from tvm.relay import transform

from . import mlp
from . import resnet
from . import dqn
from . import dcgan
from . import mobilenet
from . import lstm
from . import inception_v3
from . import squeezenet
from . import vgg
from . import densenet
from . import yolo_detection

from .config import ctx_list
from .init import create_workload
from .nat import add_nat_definitions, count, make_nat_value, make_nat_expr
from .py_converter import to_python, run_as_python


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, transform.Pass)
    mod = relay.Module.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def run_infer_type(expr):
    return run_opt_pass(expr, transform.InferType())
