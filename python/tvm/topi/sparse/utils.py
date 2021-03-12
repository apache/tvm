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
"""Some utils for Sparse operation."""


def random_bsr_matrix(m, n, bs_r, bs_c, density, dtype):
    """Generate a random sparse matrix in bsr format.

    Returns
    -------
    scipy.sparse.bsr_matrix
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import itertools
    import scipy.sparse as sp

    y = np.zeros((m, n), dtype=dtype)
    assert m % bs_r == 0
    assert n % bs_c == 0
    nnz = int(density * m * n)
    num_blocks = int(nnz / (bs_r * bs_c)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, m, bs_r), range(0, n, bs_c))))
    assert candidate_blocks.shape[0] == m // bs_r * n // bs_c
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    # pylint: disable=invalid-name
    for (r, c) in chosen_blocks:
        y[r : r + bs_r, c : c + bs_c] = np.random.randn(bs_r, bs_c)
    s = sp.bsr_matrix(y, blocksize=(bs_r, bs_c))
    assert s.data.shape == (num_blocks, bs_r, bs_c)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (m // bs_r + 1,)
    return s
