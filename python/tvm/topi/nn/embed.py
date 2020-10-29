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

"""Embedding operators"""

from tvm import te


@te.hybrid.script
def embed(table, indices):
    out = output_tensor((indices.shape[0], table.shape[1]), table.dtype)
    for i in range(indices.shape[0]):
        for j in range(table.shape[1]):
            out[i, j] = table[indices[i], j]
    return out


@te.hybrid.script
def embed_grad(table, indices, grad_in):
    grad_out = output_tensor(table.shape, table.dtype)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            grad_out[i, j] = cast(table.dtype, 0.0)
    for i in range(indices.shape[0]):
        for j in range(table.shape[1]):
            grad_out[indices[i], j] += grad_in[i, j]
    return grad_out
