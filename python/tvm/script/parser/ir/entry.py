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
"""The entry point of TVM parser for ir module."""

import inspect
from typing import Type

from tvm.ir import IRModule

from .._core import parse, utils


def ir_module(mod: Type) -> IRModule:
    """The parsing method for ir module, by using `@ir_module` as decorator.

    Parameters
    ----------
    mod : Type
        The class to be parsed as ir module.

    Returns
    -------
    ir_module : IRModule
        The parsed ir module.
    """
    if not inspect.isclass(mod):
        raise TypeError(f"Expect a class, but got: {mod}")

    return parse(mod, utils.inspect_class_capture(mod))


setattr(ir_module, "dispatch_token", "ir")
