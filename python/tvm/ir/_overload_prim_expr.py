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
"""Primitive-expression overloads for shared IR expressions."""


def __add__(_lhs, _rhs):
    return NotImplemented


def __radd__(_lhs, _rhs):
    return NotImplemented


def __sub__(_lhs, _rhs):
    return NotImplemented


def __rsub__(_lhs, _rhs):
    return NotImplemented


def __mul__(_lhs, _rhs):
    return NotImplemented


def __rmul__(_lhs, _rhs):
    return NotImplemented


def __div__(_lhs, _rhs):
    return NotImplemented


def __rdiv__(_lhs, _rhs):
    return NotImplemented


def __truediv__(_lhs, _rhs):
    return NotImplemented


def __rtruediv__(_lhs, _rhs):
    return NotImplemented


def __floordiv__(_lhs, _rhs):
    return NotImplemented


def __rfloordiv__(_lhs, _rhs):
    return NotImplemented


def __mod__(_lhs, _rhs):
    return NotImplemented


def __rmod__(_lhs, _rhs):
    return NotImplemented


def __neg__(_value):
    return NotImplemented


def __lshift__(_lhs, _rhs):
    return NotImplemented


def __rlshift__(_lhs, _rhs):
    return NotImplemented


def __rshift__(_lhs, _rhs):
    return NotImplemented


def __rrshift__(_lhs, _rhs):
    return NotImplemented


def __and__(_lhs, _rhs):
    return NotImplemented


def __rand__(_lhs, _rhs):
    return NotImplemented


def __or__(_lhs, _rhs):
    return NotImplemented


def __ror__(_lhs, _rhs):
    return NotImplemented


def __xor__(_lhs, _rhs):
    return NotImplemented


def __rxor__(_lhs, _rhs):
    return NotImplemented


def __invert__(_value):
    return NotImplemented


def __lt__(_lhs, _rhs):
    return NotImplemented


def __le__(_lhs, _rhs):
    return NotImplemented


def __eq__(_lhs, _rhs):
    return NotImplemented


def __ne__(_lhs, _rhs):
    return NotImplemented


def __gt__(_lhs, _rhs):
    return NotImplemented


def __ge__(_lhs, _rhs):
    return NotImplemented


def equal(_lhs, _rhs, _span=None):
    return NotImplemented


def astype(_value, _dtype, _span=None):
    return NotImplemented
