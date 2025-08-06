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
"""Serialization related utilities to enable some object can be pickled"""

from typing import Optional, Any
from . import _ffi_api


def to_json_graph_str(obj: Any, metadata: Optional[dict] = None):
    """
    Dump an object to a JSON graph string.

    The JSON graph string is a string representation of of the object
    graph includes the reference information of same objects, which can
    be used for serialization and debugging.

    Parameters
    ----------
    obj : Any
        The object to save.

    metadata : Optional[dict], optional
        Extra metadata to save into the json graph string.

    Returns
    -------
    json_str : str
        The JSON graph string.
    """
    return _ffi_api.ToJSONGraphString(obj, metadata)


def from_json_graph_str(json_str: str):
    """
    Load an object from a JSON graph string.

    The JSON graph string is a string representation of of the object
    graph that also includes the reference information.

    Parameters
    ----------
    json_str : str
        The JSON graph string to load.

    Returns
    -------
    obj : Any
        The loaded object.
    """
    return _ffi_api.FromJSONGraphString(json_str)


__all__ = ["from_json_graph_str", "to_json_graph_str"]
