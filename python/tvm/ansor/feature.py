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

""""Python API for Feature extraction.
The specification of features can be found in `autoscheduler_doc/per_stage_feature.md`
"""

from typing import List, Tuple
import struct
import numpy as np

from .loop_state import StateObject
from .measure import MeasureInput, MeasureResult
from . import _ffi_api


DEFAULT_MAX_N_BUFS = 5

DEFAULT_FEATURE_VEC_LEN = 164


def unpack_feature(byte_arr: bytearray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the encoded feature (in byte array format) of from c++"""
    size_of_int = 4
    size_of_float = 4

    """
    The format for n records is:
    {
      int n;
      int[n+2] sizes
      
      float[sizes[0]]    feature for record 1
      float[sizes[1]]    feature for record 2
      ...                feature for record i...
      float[sizes[n-1]]  feature for record n
      
      float[sizes[n]]    normalized throughput for n records
      int[sizes[n+1]]    task id for n records
    }
    """
    vec_len = DEFAULT_FEATURE_VEC_LEN

    # unpack sizes
    offset = 0
    n = struct.unpack_from("1i", byte_arr, offset=offset)[0]
    offset += size_of_int

    sizes = struct.unpack_from("%di" % (n+2), byte_arr, offset=offset)
    offset += size_of_int * (n+2)

    # unpack features
    features = []
    for size in sizes[:-2]:
        row = []

        """
        Now we need to unpack the feature for multiple statements.
        The format is:
        {
            int n_stmts
            float[n_stmt][vec_len] feature_vecs
        }
        where vec_len can be calculated by `(size - 1) / n_stmts`
        """
        if size == 0:
            # failed during lowering
            features.append(np.zeros((1, vec_len)))
        else:
            n_stmts = struct.unpack_from("f", byte_arr, offset=offset)
            offset += size_of_float

            n_stmts = int(n_stmts[0] + 0.5)
            tmp_vec_len = (size - 1) // n_stmts
            assert tmp_vec_len == vec_len, "The lenght of feature vector is wrong. " \
                                           "Expected %d but got %d." % (vec_len, tmp_vec_len)
            assert (size - 1) % n_stmts == 0
            for _ in range(n_stmts):
                x = struct.unpack_from("%df" % vec_len, byte_arr, offset=offset)
                offset += vec_len * size_of_float
                row.append(x)

            features.append(np.array(row))

    # unpack normalized_throughputs
    m = sizes[-2]
    normalized_throughputs = struct.unpack_from("%df" % m, byte_arr, offset=offset)
    offset += m * size_of_int

    # unpack task_ids
    m = sizes[-1]
    task_ids = struct.unpack_from("%di" % m, byte_arr, offset=offset)
    offset += m * size_of_int

    assert offset == len(byte_arr), "%d vs %d" % (offset, len(byte_arr))
    return np.array(features), np.array(normalized_throughputs), np.array(task_ids)


def get_per_stmt_features_from_file(filename: str,
                                    n_lines: int,
                                    max_n_bufs: int = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per_stmt features from a log file"""
    byte_arr = _ffi_api.GetPerStmtFeaturesFromFile(
        filename, n_lines, max_n_bufs or DEFAULT_MAX_N_BUFS)
    return unpack_feature(byte_arr)


def get_per_stmt_features_from_measure_pairs(inputs: List[MeasureInput],
                                             results: List[MeasureResult],
                                             skip_first_n_feature_extraction: int = 0,
                                             max_n_bufs: int = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per_stmt features from measurement pairs"""
    byte_arr = _ffi_api.GetPerStmtFeaturesFromMeasurePairs(
        inputs, results, skip_first_n_feature_extraction, max_n_bufs or DEFAULT_MAX_N_BUFS)
    return unpack_feature(byte_arr)


def get_per_stmt_features_from_states(states: List[StateObject],
                                      task: "SearchTask",
                                      max_n_bufs: int = None) -> List[np.ndarray]:
    """Get per_stmt features from states"""
    byte_arr = _ffi_api.GetPerStmtFeaturesFromStates(
        states, task, max_n_bufs or DEFAULT_MAX_N_BUFS)
    return unpack_feature(byte_arr)[0]


def get_per_stmt_feature_names(max_n_bufs: int = None) -> List[str]:
    """Get names of the elements in the flatten feature vector"""
    return [x for x in
            _ffi_api.GetPerStmtFeatureNames(max_n_bufs or DEFAULT_MAX_N_BUFS)]
