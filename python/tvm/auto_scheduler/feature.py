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

""""
Python API for Feature extraction. The extracted features vector are used by cost models.

We extract one feature vector per BufferStoreNode statement in a TIR Stmt,
so we call this feature as "per-store" feature.
The cost model also does prediction for each BufferStoreNode statement and aggregates
the predicted score of each BufferStoreNode as the score of a TIR Stmt.

The feature specification is defined by `src/auto_scheduler/feature.cc::FeatureSet`
"""

from typing import List, Tuple, Union, Optional
import struct

import numpy as np

from .loop_state import State, StateObject
from .measure import MeasureInput, MeasureResult
from . import _ffi_api

# The maximum number of extracted buffers for one statement
DEFAULT_MAX_N_BUFS = 5

# The length of the feature vector
DEFAULT_FEATURE_VEC_LEN = 164

# The size of int and float in bytes
SIZE_OF_INT32 = 4
SIZE_OF_FLOAT32 = 4


def unpack_feature(byte_arr: bytearray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the flatten feature (in byte array format) from c++

    Parameters
    ----------
    byte_arr: bytearray
        The two-dimensional feature vector in serialized byte array format

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids

    Note
    ----
    For faster data copy between c++ and python, the c++ part returns features in a single
    flatten array using a packed format. The python part then unpacks the flatten array.

    The packed format for n records is:
    {
      int   n;
      int   sizes[n+2];           // The sizes for the following arrays

      float features_0[size[0]];  // The features for record 0
      float features_1[size[1]];  // The features for record 1
      ...
      float features_i[size[i]];  // The features for record i
      ... // until i == n - 1

      float throughputs[sizes[n]];  // The normalized throughputs for n records
      int   task_ids[size[n+1]];    // The task ids for n records

    }
    To implement this format, we also store int as float, so we can store all numbers
    into a single float array.
    """
    vec_len = DEFAULT_FEATURE_VEC_LEN

    # unpack sizes
    offset = 0
    n = struct.unpack_from("1i", byte_arr, offset=offset)[0]
    offset += SIZE_OF_INT32

    sizes = struct.unpack_from("%di" % (n + 2), byte_arr, offset=offset)
    offset += SIZE_OF_INT32 * (n + 2)

    # unpack features
    features = []
    for size in sizes[:-2]:
        row = []

        # Now, we need to unpack the feature for multiple statements.
        # The format is:
        # {
        #   int   n_stage;                        // The number of stages
        #   float feature_vecs[n_stage][vec_len]  // The feature vector for each stage
        # }
        # where vec_len can be calculated by `(size - 1) / n_stmts`

        if size == 0:
            # failed during lowering
            features.append(np.zeros((1, vec_len)))
        else:
            n_stmts = struct.unpack_from("f", byte_arr, offset=offset)
            offset += SIZE_OF_FLOAT32

            n_stmts = int(n_stmts[0] + 0.5)
            tmp_vec_len = (size - 1) // n_stmts
            assert (
                tmp_vec_len == vec_len
            ), "The length of feature vector is wrong. Expected %d but got %d." % (
                vec_len,
                tmp_vec_len,
            )
            assert tmp_vec_len * n_stmts == size - 1
            for _ in range(n_stmts):
                x = struct.unpack_from("%df" % vec_len, byte_arr, offset=offset)
                offset += vec_len * SIZE_OF_FLOAT32
                row.append(x)

            features.append(np.array(row))

    # unpack normalized_throughputs
    m = sizes[-2]
    normalized_throughputs = struct.unpack_from("%df" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_FLOAT32

    # unpack task_ids
    m = sizes[-1]
    task_ids = struct.unpack_from("%di" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_INT32

    assert offset == len(byte_arr), "%d vs %d" % (offset, len(byte_arr))
    return np.array(features, dtype=object), np.array(normalized_throughputs), np.array(task_ids)


def get_per_store_features_from_file(
    filename: str, max_lines: int, max_n_bufs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per-store features from a log file

    Parameters
    ----------
    filename: str
        The input filename
    max_lines: int
        Only extract the first n lines of the file
    max_n_bufs: Optional[int]
        The maximum number of extracted buffers for one statement

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids
    """
    byte_arr = _ffi_api.GetPerStoreFeaturesFromFile(
        filename, max_lines, max_n_bufs or DEFAULT_MAX_N_BUFS
    )
    return unpack_feature(byte_arr)


def get_per_store_features_from_measure_pairs(
    inputs: List[MeasureInput],
    results: List[MeasureResult],
    skip_first_n_feature_extraction: int = 0,
    max_n_bufs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per-store features from measurement input/result pairs

    Parameters
    ----------
    inputs: List[MeasureInput]
        The measure inputs
    results: List[MeasureResult]
        The measure results
    skip_first_n_feature_extraction: int
        Skip feature extraction for the first n states
    max_n_bufs: int
        The maximum number of extracted buffers for one statement

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids
    """
    byte_arr = _ffi_api.GetPerStoreFeaturesFromMeasurePairs(
        inputs, results, skip_first_n_feature_extraction, max_n_bufs or DEFAULT_MAX_N_BUFS
    )
    return unpack_feature(byte_arr)


def get_per_store_features_from_states(
    states: List[Union[State, StateObject]], task: "SearchTask", max_n_bufs: Optional[int] = None
) -> np.ndarray:
    """Get per-store features from measurement input/result pairs

    Parameters
    ----------
    states: List[Union[State, StateObject]]
        The input states
    task: SearchTask
        The search task of the input states
    max_n_bufs: Optional[int]
        The maximum number of extracted buffers for one statement

    Returns
    -------
    features: np.ndarray
        Feature vectors
    """
    if isinstance(states[0], State):
        state_objects = [s.state_object for s in states]
    elif isinstance(states[0], StateObject):
        state_objects = states
    byte_arr = _ffi_api.GetPerStoreFeaturesFromStates(
        state_objects, task, max_n_bufs or DEFAULT_MAX_N_BUFS
    )
    return unpack_feature(byte_arr)[0]


def get_per_store_feature_names(max_n_bufs: Optional[int] = None) -> List[str]:
    """Get the name of every element in the feature vector. Use this for debug and inspection.

    Parameters
    ----------
    max_n_bufs: int
        The maximum number of extracted buffers for one statement

    Returns
    -------
    names: List[str]
        The names of elements in the flatten feature vector
    """
    return _ffi_api.GetPerStoreFeatureNames(max_n_bufs or DEFAULT_MAX_N_BUFS)
