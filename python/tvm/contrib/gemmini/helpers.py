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
"""
Miscellaneous helpers
=====================
"""

from typing import List
from six.moves import range
from .environment import Environment


ENV = Environment.instance()


def get_divisors(x: int) -> List[int]:
    """Gets all the numbers that perfectly divide x

    Args:
        x (int): Number to divide

    Returns:
        List[int]: list of divisors
    """
    divs = []
    for i in range(1, x + 1):
        if x % i == 0:
            divs.append(i)
    return divs


def get_greater_div(x, limit: int = None):
    """Gets the greater divisor for all x

    Args:
        x: _description_
        limit (int, optional): Max greater divisor to return. Defaults to None.

    Returns:
        int: Greater divisor
    """

    limit = ENV.DIM if limit is None else limit

    if isinstance(x, int):
        elements = [x]
    elif isinstance(x, list):
        elements = x
    else:
        assert False, "datatype of x not supported!"

    divisors = []
    for element in elements:
        divs = get_divisors(element)
        filtered = filter(lambda d: d <= limit, divs)
        divisors.append(filtered)

    return max(set.intersection(*map(set, divisors)))
