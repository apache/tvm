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
from typing import List, Type
from types import FrameType

from tvm.ir import IRModule

from .._core import parse, utils


def is_defined_in_class(frames: List[FrameType]) -> bool:
    """Check whether a object is defined in a class scope.

    Parameters
    ----------
    frames : List[FrameType]
        The frame stack of the object, obtained by `inspect.stack()`.

    Returns
    -------
    res : bool
        The result if the object is defined in a class scope.
    """
    if len(frames) > 2:
        maybe_class_frame = frames[2]
        statement_list = maybe_class_frame[4]
        if statement_list is None:
            return False
        first_statement = statement_list[0]
        line = first_statement.strip()
        if line.startswith("class "):
            return True
        if line.startswith("@") and "ir_module" in line:
            return True
    return False


def ir_module(mod: Type) -> IRModule:
    """The parsing method for ir module, by using `@ir_module` as decorator.

    Parameters
    ----------
    mod : Type
        The class to be parsed as ir module.

    Returns
    -------
    irmodule : IRModule
        The parsed ir module.
    """
    if not inspect.isclass(mod):
        raise TypeError(f"Expect a class, but got: {mod}")

    return parse(mod, utils.inspect_class_capture(mod))


setattr(ir_module, "dispatch_token", "ir")
