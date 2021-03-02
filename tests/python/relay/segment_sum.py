import torch
import numpy as np


def segment_sum_torch(src, indices, data):
    indices = indices.expand([2, 1, -1]).permute(2, 1, 0)
    print(indices)
    result = src.scatter_add(0, indices, data)
    return result


def segment_sum_numpy(src, indices, data):
    for i, index in enumerate(indices):
        src[index] += data[i]

    return src


if __name__ == "__main__":
    data = np.array([[[1, 7]], [[3, 8]], [[2, 9]]], dtype=np.float32)
    src = np.zeros((2, 1, 2), dtype=np.float32)
    indices = np.array([0, 0, 1], dtype=np.int64)
    result_torch = segment_sum_torch(
        torch.from_numpy(src), torch.from_numpy(indices), torch.from_numpy(data)
    )
    result_numpy = segment_sum_numpy(src, indices, data)
    np.testing.assert_allclose(result_torch, result_numpy)