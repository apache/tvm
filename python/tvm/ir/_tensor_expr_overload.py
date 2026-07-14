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
"""Tensor-expression overload hooks for shared IR expressions."""


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


def __pow__(_lhs, _rhs):
    return NotImplemented


def __rpow__(_lhs, _rhs):
    return NotImplemented


def __neg__(_value):
    return NotImplemented


def __lt__(_lhs, _rhs):
    return NotImplemented


def __le__(_lhs, _rhs):
    return NotImplemented


def __gt__(_lhs, _rhs):
    return NotImplemented


def __ge__(_lhs, _rhs):
    return NotImplemented


def __call__(_value, *_args, **_kwargs):
    return NotImplemented


def __getitem__(_value, _index):
    return NotImplemented


def astype(_value, _dtype, _span=None):
    return NotImplemented
