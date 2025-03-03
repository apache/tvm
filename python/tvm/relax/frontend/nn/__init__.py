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
"""A PyTorch-like API to build IRModules."""
# pylint: disable=redefined-builtin
from . import op, spec
from .core import Effect, Module, ModuleList, Object, Parameter, Tensor
from .exporter import add_extern
from .extern import ExternModule, ObjectModule, SourceModule
from .modules import (
    GELU,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose1D,
    Embedding,
    GroupNorm,
    IOEffect,
    KVCache,
    LayerNorm,
    Linear,
    ReLU,
    RMSNorm,
    SiLU,
)
from .op import *
from .subroutine import SubroutineMixin
from .visitor import Mutator
