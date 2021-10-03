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
"""TVM Script Parser Function Registry """
# pylint: disable=inconsistent-return-statements, relative-beyond-top-level, import-outside-toplevel
import types
from typing import Union, Callable, Dict, Optional, Any


class Registry(object):
    """Registration map
    All these maps are static
    """

    registrations: Dict[str, type] = dict()

    @staticmethod
    def lookup(name: str) -> Optional[Any]:
        if name in Registry.registrations:
            # every time we create a new handler
            # since we may want to keep some local info inside it
            return Registry.registrations[name]()
        return None


def register(inputs: Union[Callable, type]) -> type:
    """Register Intrin/ScopeHandler/SpecialStmt"""
    registration: type
    if isinstance(inputs, types.FunctionType):
        # is function
        from .tir.intrin import Intrin

        def create_new_intrin(func) -> type:
            class NewIntrin(Intrin):
                def __init__(self):
                    super().__init__(func)

            return NewIntrin

        registration = create_new_intrin(inputs)
    elif isinstance(inputs, type):
        # is class
        registration = inputs
    else:
        raise ValueError()

    key: str = registration().signature()[0]
    Registry.registrations[key] = registration
    return registration
