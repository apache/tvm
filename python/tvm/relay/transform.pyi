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

import tvm
from .base import NodeBase


class PassContext(NodeBase):
    def __init__(self):
        ...

class PassInfo(NodeBase):
    name = ...  # type: str
    opt_level = ... # type: int
    required = ... # type: list

    def __init__(self, name, opt_level, required)
        # type: (str, int, list) -> None


class Pass(NodeBase):
    def __init__(self):
        ...


class ModulePass(Pass):
    name = ...  # type: str
    opt_level = ...  # type: int
    pass_func = ...  # type: Callable
    required = ...  # type: list

    def __init__(self, name, opt_level, pass_func, required):
        # type: (str, int, Callable, list) -> None
        ...


class FunctionPass(Pass):
    name = ...  # type: str
    opt_level = ...  # type: int
    pass_func = ...  # type: Callable
    required = ...  # type: list

    def __init__(self, name, opt_level, pass_func, required):
        # type: (str, int, Callable, list) -> None
        ...


class Sequential(Pass):
    name = ...  # type: str
    opt_level = ...  # type: int
    passes = ...  # type: list
    required = ...  # type: list
    disabled = ... # type: list

    def __init__(self, name, opt_level, passes, required, disabled):
        # type: (str, int, list, list, list) -> None
        ...
