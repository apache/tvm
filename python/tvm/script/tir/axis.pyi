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
# pylint: disable=redefined-builtin
from typing import Tuple, Union, List
from tvm.tir import PrimExpr, IterVar, Var

def spatial(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
def S(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], iter_value) -> IterVar: ...
def reduce(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
def R(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
def scan(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
def opaque(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
def remap(iter_types: str, loop_vars: List[Var]) -> IterVar: ...