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
"""Testing utilities."""

from . import _ffi_api
from .core import Object
from .registry import register_object


@register_object("testing.TestObjectBase")
class TestObjectBase(Object):
    """
    Test object base class.
    """


@register_object("testing.TestObjectDerived")
class TestObjectDerived(TestObjectBase):
    """
    Test object derived class.
    """


def create_object(type_key: str, **kwargs) -> Object:
    """
    Make an object by reflection.

    Parameters
    ----------
    type_key : str
        The type key of the object.
    kwargs : dict
        The keyword arguments to the object.

    Returns
    -------
    obj : object
        The created object.

    Note
    ----
    This function is only used for testing purposes and should
    not be used in other cases.
    """
    args = [type_key]
    for k, v in kwargs.items():
        args.append(k)
        args.append(v)
    return _ffi_api.MakeObjectFromPackedArgs(*args)
