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
"""Primitive expression nodes shared by TVM IR dialects."""

_EXPR_NAMES = {
    "StringImm",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "FloorDiv",
    "FloorMod",
    "Min",
    "Max",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "And",
    "Or",
    "Not",
    "Select",
    "Let",
    "Ramp",
    "Broadcast",
    "Shuffle",
}


def __getattr__(name):
    # Keep the historical tvm.tirx classes as the single Python definitions
    # during this mechanical C++ ownership move.  Importing lazily avoids the
    # tvm.ir <-> tvm.tirx initialization cycle while exposing the new public
    # tvm.ir.prim spelling.
    if name in _EXPR_NAMES:
        from tvm.tirx import expr  # pylint: disable=import-outside-toplevel

        return getattr(expr, name)
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | _EXPR_NAMES)
