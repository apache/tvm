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
"""Common utility functions for CUDA op codegen."""


def parse_str(arg) -> str:
    """Parse TIR StringImm or Python str to a plain str.

    TIR StringImm values stringify to quoted strings, e.g., ``'"float16"'``;
    Python strs do not. Idempotent — passing an already-parsed str returns it
    unchanged, so dispatchers that parse once before forwarding to inner
    codegens won't double-strip the value.
    """
    s = str(arg)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def validate_cta_group(cta_group, context: str = "") -> int:
    """Validate that cta_group is 1 or 2 and return it as int.

    Args:
        cta_group: The cta_group value (can be int or TIR IntImm)
        context: Optional context string for error message (e.g., "allocating Tensor Memory")

    Returns:
        The validated cta_group as int

    Raises:
        ValueError: If cta_group is not 1 or 2
    """
    cta_group = int(cta_group)
    if cta_group not in [1, 2]:
        ctx = f" involved in {context}" if context else ""
        raise ValueError(
            f"The number of cta_group{ctx} is incorrect, expected 1 or 2, got {cta_group}"
        )
    return cta_group


def validate_power_of_two_range(value, min_val: int, max_val: int, name: str) -> int:
    """Validate that value is within range and is a power of two.

    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Returns:
        The validated value as int

    Raises:
        ValueError: If value is out of range or not a power of two
    """
    value = int(value)
    if not (min_val <= value <= max_val and is_power_of_two(value)):
        raise ValueError(
            f"The {name} is invalid, expect a value within range [{min_val}, {max_val}] "
            f"and be a power of 2, got {value}"
        )
    return value
