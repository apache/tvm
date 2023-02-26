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
import tvm
import tvm.testing
from tvm import topi
import tvm.topi.testing


@tvm.testing.parametrize_targets
def test_scatter_nd(dev, target):
    def check_scatter_nd(data, indices, updates, out, mode="add"):
        implementations = {
            "generic": (
                lambda x, y, z: topi.scatter_nd(x, y, z, mode),
                topi.generic.schedule_extern,
            ),
            "gpu": (
                lambda x, y, z: topi.cuda.scatter_nd(x, y, z, mode),
                topi.generic.schedule_extern,
            ),
        }
        fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)
        tvm.topi.testing.compare_numpy_tvm(
            [data, indices, updates], out, target, dev, fcompute, fschedule
        )

    data = np.zeros((2, 2)).astype("int64")
    indices = np.array([[1, 1, 0], [0, 1, 0]])
    updates = np.array([2, 3, 0])
    out = np.array([[0, 0], [2, 3]])
    check_scatter_nd(data, indices, updates, out)

    data = np.zeros((2, 2, 2, 2)).astype("int64")
    indices = np.array([[0, 1], [1, 1]])
    updates = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    out = np.array([[[[0, 0], [0, 0]], [[1, 2], [3, 4]]], [[[0, 0], [0, 0]], [[5, 6], [7, 8]]]])
    check_scatter_nd(data, indices, updates, out)

    indices = np.array([[1, 0, 0]])
    updates = np.reshape(np.arange(1560 * 3), (3, 1560)).astype("float32")
    shape = (2, 1560)
    data = np.zeros(shape).astype("float32")
    out = data.copy()
    out[1, :] += updates[0, :]
    out[0, :] += updates[1, :]
    out[0, :] += updates[2, :]
    check_scatter_nd(data, indices, updates, out)

    for mode in ["update", "add", "mul", "min", "max"]:
        updates = np.ones((5, 3)).astype("float64")
        indices = np.stack((np.random.randint(2, size=5), np.random.randint(7, size=5))).astype(
            "int64"
        )
        shape = (2, 7, 3)
        data = np.random.random(shape).astype("float64")
        out = data.copy()
        for i in range(indices.shape[1]):
            for j in range(updates.shape[1]):
                if mode == "update":
                    out[indices[0, i], indices[1, i], j] = updates[i, j]
                elif mode == "add":
                    out[indices[0, i], indices[1, i], j] += updates[i, j]
                elif mode == "mul":
                    out[indices[0, i], indices[1, i], j] *= updates[i, j]
                elif mode == "min":
                    out[indices[0, i], indices[1, i], j] = min(
                        out[indices[0, i], indices[1, i], j], updates[i, j]
                    )
                elif mode == "max":
                    out[indices[0, i], indices[1, i], j] = max(
                        out[indices[0, i], indices[1, i], j], updates[i, j]
                    )

        check_scatter_nd(data, indices, updates, out, mode)


if __name__ == "__main__":
    test_scatter_nd(tvm.device("cpu"), tvm.target.Target("llvm"))
