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
from typing import List
import numpy as np


def get_all_factors(n: int) -> List[int]:
    # Calculate the square root of n and round it up to the nearest integer
    n0 = int(np.ceil(np.sqrt(n)))

    # Find all divisors of n that are less than n0
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1

    # If n is a perfect square, add the square root to the list of factors
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]

    # Combine the factors and their corresponding larger pair factors
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]


def factorize(n: int) -> List[int]:
    i = 2  # Start with the smallest prime number
    result = []

    # Iterate through numbers to find factors
    while n > 1:
        if n % i == 0:  # If i is a factor of n
            n //= i  # Divide n by i and keep the integer part
            result.append(i)
        else:
            i += 1  # Try the next number
    return result


def coalesced_factor(subtensor: List[int], tensor: List[int]) -> int:
    # If the last dimension of the subtensor and tensor differ, or subtensor has only one dimension
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        # Recursively calculate the coalesced factor for the remaining dimensions
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])


def coalesced_tensor_shape(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    # Calculate the total number of elements in the subtensor
    bytes = int(np.prod(subtensor))

    if bytes == 0:
        return 0

    # Calculate the coalesced factor for the subtensor
    factor = int(coalesced_factor(subtensor, tensor))

    # Compute the shape of the coalesced tensor
    return transaction_size * bytes / min(transaction_size, factor)
