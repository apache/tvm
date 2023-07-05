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
Name transformation functions for use in code generation
"""

from typing import List, Union

from tvm import TVMError
from . import _backend


def to_c_function_style(original_name: str):
    """Transform a name to the C function style assuming it is
    appropriately constructed using the prefixing functions

    Parameters
    ----------
    original_name : str
        Original name to transform
    """
    return _backend.ToCFunctionStyle(original_name)


def to_c_variable_style(original_name: str):
    """Transform a name to the C variable style assuming it is
    appropriately constructed using the prefixing functions

    Parameters
    ----------
    original_name : str
        Original name to transform
    """
    return _backend.ToCVariableStyle(original_name)


def to_c_constant_style(original_name: str):
    """Transform a name to the C constant style assuming it is
    appropriately constructed using the prefixing functions

    Parameters
    ----------
    original_name : str
        Original name to transform
    """
    return _backend.ToCConstantStyle(original_name)


def _preprocess_names(names: Union[List[str], str]):
    """Preprocesses name strings into format for C++ functions

    Parameters
    ----------
    names : Union[List[str], str]
        List of names to combine to form a combined name or the name itself
    """
    if isinstance(names, str):
        if names == "":
            raise TVMError("Name is empty")
        return [names]
    return names


def prefix_name(names: Union[List[str], str]):
    """Apply TVM-specific prefix to a function name

    Parameters
    ----------
    names : Union[List[str], str]
        List of names to combine to form a combined name or the name itself
    """

    return _backend.PrefixName(_preprocess_names(names))


def prefix_generated_name(names: Union[List[str], str]):
    """Apply generated TVM-specific prefix to a function name

    Parameters
    ----------
    names : Union[List[str], str]
        List of names to combine to form a combined name or the name itself
    """

    return _backend.PrefixGeneratedName(_preprocess_names(names))
