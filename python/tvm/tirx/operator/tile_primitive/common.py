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
"""TIRx operator dispatch common utilities."""

from enum import Enum


class MapOpType(Enum):
    """Enumeration of common unary and binary operator types."""

    ADD = 0
    SUB = 1
    MUL = 2
    FDIV = 3
    ZERO = 4
    SQRT = 5
    RECIPROCAL = 6
    FILL = 7
    MAX = 8
    MIN = 9
    EXP = 10
    EXP2 = 11
    SILU = 12


class ReduceOpType(Enum):
    """Enumeration of common reduce operator types."""

    SUM = 0
    MAX = 1
    MIN = 2
