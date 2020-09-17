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
"""Hybrid Script Parser Typing Class

This module provides typing class for hybrid script type annotation usage, it can be viewed as
a wrapper for uniform Type system in IR
"""
# pylint: disable=invalid-name
import tvm


class TypeGeneric:
    """Base class for all the hybrid script typing class"""

    def evaluate(self):
        raise TypeError("Cannot get tvm.Type from a generic type")


class ConcreteType(TypeGeneric):
    """Hybrid script typing class for uniform Type objects"""

    def __init__(self, vtype):
        self.type = vtype

    def evaluate(self):
        return tvm.ir.PrimType(self.type)


class GenericPtrType(TypeGeneric):
    """Hybrid script typing class generator for PtrType

    [] operator is overloaded, accepts a ConcreteType and returns a ConcreteType wrapping PtrType
    """

    def __getitem__(self, vtype):
        return ConcreteType(tvm.ir.PointerType(vtype.evaluate()))


class GenericTupleType(TypeGeneric):
    """Hybrid script typing class generator for TupleType

    [] operator is overloaded, accepts a list of ConcreteType and returns a ConcreteType
    wrapping TupleType
    """

    def __getitem__(self, vtypes):
        return ConcreteType(tvm.ir.TupleType([vtype.evaluate() for vtype in vtypes]))


int32 = ConcreteType("int32")
handle = ConcreteType("handle")
Ptr = GenericPtrType()
Tuple = GenericTupleType()
