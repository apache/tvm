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

from tvm.relay import Var, TypeVar
from typing import Any, Optional, List, Tuple
import attr

class LittleCppNode:
    pass

@attr.s(auto_attribs=True)
class Decl(LittleCppNode):
    bindings: List[Tuple[Var, LittleCppNode]]
    body: LittleCppNode

@attr.s(auto_attribs=True)
class PackedCall(LittleCppNode):
    name: str
    args: Any
    args_type: Any
    ret_type: Any

@attr.s(auto_attribs=True)
class Invoke(LittleCppNode):
    call: Any
    args: Any

@attr.s(auto_attribs=True)
class CPPFunction(LittleCppNode):
    params: List[Var]
    body: Any
    ret_type: Any
    name: Optional[str] = None

@attr.s(auto_attribs=True)
class CPPIf(LittleCppNode):
    cond: Any
    true_branch: Any
    false_branch: Any
    relay_type: Any

@attr.s(auto_attribs=True)
class CPPTuple(LittleCppNode):
    fields: List[Any]
    relay_type: Any

@attr.s(auto_attribs=True)
class CPPMatch(LittleCppNode):
    data: Any
    clause: List[Tuple[Any, Any]]
    relay_type: Any

@attr.s(auto_attribs=True)
class CPPConstructor(LittleCppNode):
    tag: int
    fields: List[Any]

@attr.s(auto_attribs=True)
class CPPTupleGetItem(LittleCppNode):
    tuple_value: Any
    index: int
    relay_type: Any

@attr.s(auto_attribs=True)
class CPPRefCreate(LittleCppNode):
    value: Any
    relay_type: Any

@attr.s(auto_attribs=True)
class CPPRefRead(LittleCppNode):
    ref: Any
    relay_type: Any

@attr.s(auto_attribs=True)
class CPPRefWrite(LittleCppNode):
    ref: Any
    value: Any
