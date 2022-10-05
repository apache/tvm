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
"""TVM Script Parser utils"""

import inspect
from typing import Any, Callable, Dict


def inspect_function_capture(func: Callable) -> Dict[str, Any]:
    """Capture function non-locals and global variables.

    Parameters
    ----------
    func : Callable
        The function to inspect.

    Returns
    -------
    res : Dict[str, Any]
        The function variables map with non-local or global variables.
    """
    captured = {
        **inspect.getclosurevars(func).nonlocals,
        **func.__globals__,  # type: ignore
    }
    return captured


def inspect_class_capture(cls: type) -> Dict[str, Any]:
    """Capture class non-locals and global variables.

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    res : Dict[str, Any]
        The class variables map with non-local or global variables.
    """
    result: Dict[str, Any] = {}
    for _, v in cls.__dict__.items():
        if inspect.isfunction(v):
            func_vars = inspect_function_capture(v)
            result.update(**func_vars)
    return result
