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

# pylint: disable=invalid-name
"""Testing utilities for runtime builtin functions."""
from enum import IntEnum


class MatchShapeCode(IntEnum):
    """Code passed to match shape builtin"""

    ASSERT_EQUAL_TO_IMM = 0
    STORE_TO_HEAP = 1
    NO_OP = 2
    ASSERT_EQUAL_TO_LOAD = 3


class MakeShapeCode(IntEnum):
    """Code passed to match shape builtin"""

    USE_IMM = 0
    LOAD_SHAPE = 1
