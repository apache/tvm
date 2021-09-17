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
import numpy as np


def gather_nd(data_np, indices_np, batch_dims=0):
    """gather_nd implemented using numpy"""
    data_shape = data_np.shape
    indices_shape = indices_np.shape

    def gather_nd_batch_dims_1_ref(data, indices):
        res = []
        for i, row in enumerate(data):
            indices_tuple = tuple(indices[:, i])  # the indices for the i-th batch
            res.append(row[indices_tuple])
        # stack on the batch dim
        return np.stack(res, 0)

    if batch_dims > 1:
        data_np_reshape = np.reshape(data_np, (-1,) + data_shape[batch_dims:])
        indices_np_reshape = np.reshape(
            indices_np, (indices_shape[0], -1) + indices_shape[(batch_dims + 1) :]
        )

        ref_res = gather_nd_batch_dims_1_ref(data_np_reshape, indices_np_reshape)

        out_shape = indices_shape[1 : (batch_dims + 1)] + ref_res.shape[1:]
        ref_res = np.reshape(ref_res, out_shape)
    elif batch_dims == 1:
        ref_res = gather_nd_batch_dims_1_ref(data_np, indices_np)
    else:
        ref_res = data_np[tuple(indices_np)]

    return ref_res
