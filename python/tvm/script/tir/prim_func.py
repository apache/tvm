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
"""TVM Script Interface for PrimFunc"""

import inspect
from typing import Callable

from tvm.tir.function import PrimFunc
from ..parser import from_source


def prim_func(input_func: Callable) -> PrimFunc:
    """Decorate a python function as tvm script.

    Parameters
    ----------
    func : input_func
        The function to be parsed.

    Returns
    -------
    output : PrimFunc
        The result functions.
    """
    if inspect.isfunction(input_func):
        result = from_source(input_func)
        result.__name__ = input_func.__name__
        result.__qualname__ = input_func.__qualname__
        return result

    raise TypeError("Only function definitions are supported.")
