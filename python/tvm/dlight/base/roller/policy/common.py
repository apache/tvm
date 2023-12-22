from typing import List

import numpy as np


def get_all_factors(n: int) -> List[int]:
    n0 = int(np.ceil(np.sqrt(n)))
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]

def factorize(n: int) -> List[int]:
    i = 2
    result = []
    while n > 1:
        if n % i == 0:
            n //= i
            result.append(i)
        else:
            i += 1
    return result

def coalesced_factor(subtensor: List[int], tensor: List[int]) -> int:
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])

def coalesced_tensor_shape(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    bytes = int(np.prod(subtensor))
    if bytes == 0: return 0
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * bytes / min(transaction_size, factor)
