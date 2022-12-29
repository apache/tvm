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
# pylint: disable=invalid-name, no-value-for-parameter
"""Defines common C code for all microkernel operations."""


common_includes = """

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>

"""

MICRO_WORD_LENGTH_BITS = 32


def num_simd_lanes_per_word(dtype: str) -> int:
    """Takes a dtype, and returns how many of that dtype fit into a single microcontroller word.

    >>> num_simd_lanes_per_word("int8")
    4
    >>> num_simd_lanes_per_word("int16")
    2
    """
    assert dtype.startswith("int")
    dtype_width = int(dtype[3:])
    return MICRO_WORD_LENGTH_BITS // dtype_width
