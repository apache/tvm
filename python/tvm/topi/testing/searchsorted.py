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
"""The reference implementation of searchsorted in Numpy."""
import numpy as np


def searchsorted_ref(sorted_sequence, values, right, out_dtype):
    """Run Numpy searchsorted on 1-D or N-D sorted_sequence."""
    side = "right" if right else "left"
    if len(sorted_sequence.shape) == 1 and len(values.shape) > 1:
        sorted_sequence_2d = np.tile(sorted_sequence, (np.prod(values.shape[:-1]), 1))
    else:
        sorted_sequence_2d = np.reshape(sorted_sequence, (-1, sorted_sequence.shape[-1]))

    values_2d = np.reshape(values, (-1, values.shape[-1]))
    indices = np.zeros(values_2d.shape, dtype=out_dtype)

    for i in range(indices.shape[0]):
        indices[i] = np.searchsorted(sorted_sequence_2d[i], values_2d[i], side=side)

    return np.reshape(indices, values.shape)
