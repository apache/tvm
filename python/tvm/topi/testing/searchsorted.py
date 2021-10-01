import numpy as np


def searchsorted_ref(sorted_sequence, values, side, out_dtype):
    sorted_sequence_2d = np.reshape(sorted_sequence, (-1, sorted_sequence.shape[-1]))
    values_2d = np.reshape(values, (-1, values.shape[-1]))
    indices = np.zeros(values_2d.shape, dtype=out_dtype)

    for i in range(indices.shape[0]):
        indices[i] = np.searchsorted(sorted_sequence_2d[i], values_2d[i], side=side)

    return np.reshape(indices, values.shape)
