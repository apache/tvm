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
"""Test code for loss operators."""
import numpy as np
import pytest
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing

import tvm.testing


prediction_shape, reduction, ignore_index, dtype = tvm.testing.parameters(
    ((10, 5), "mean", -100, "float32"),
    ((10, 5, 2, 2), "mean", -100, "float32"),
    ((10, 5), "sum", -100, "float32"),
    ((10, 5), "none", -100, "float32"),
    ((10, 5), "mean", 3, "float32"),
    ((10, 5), "mean", -100, "float64"),
    ((5,), "mean", -100, "float32"),
    ((5,), "mean", 3, "float32"),
    ((5,), "none", -100, "float32"),
)


def test_nll_loss(target, dev, prediction_shape, reduction, ignore_index, dtype):
    if len(prediction_shape) == 1:
        C = prediction_shape[0]
        target_shape = []
    else:
        C = prediction_shape[1]
        target_shape = prediction_shape[:1] + prediction_shape[2:]
    predictions = te.placeholder(shape=prediction_shape, name="predictions", dtype=dtype)
    targets = te.placeholder(shape=target_shape, name="targets", dtype="int32")
    weights = te.placeholder(shape=(C,), name="weights", dtype=dtype)
    nll_loss_result = topi.nn.nll_loss(predictions, targets, weights, reduction, ignore_index)

    with tvm.target.Target(target):
        fschedule = tvm.topi.testing.get_reduce_schedule(target)
        s = fschedule([nll_loss_result])
    fn = tvm.build(s, [predictions, targets, weights, nll_loss_result], target, name="nll_loss")

    predictions_npy = np.random.uniform(size=prediction_shape).astype(dtype)
    targets_npy = np.random.randint(0, C, target_shape).astype("int32")
    weights_npy = np.random.uniform(size=(C,)).astype(dtype)
    out_npy = tvm.topi.testing.nll_loss(
        predictions_npy, targets_npy, weights_npy, reduction, ignore_index
    )

    predictions_nd = tvm.nd.array(predictions_npy, dev)
    targets_nd = tvm.nd.array(targets_npy, dev)
    weights_nd = tvm.nd.array(weights_npy, dev)
    out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(nll_loss_result.dtype), dev)
    fn(predictions_nd, targets_nd, weights_nd, out_nd)
    out_topi = out_nd.numpy()
    tvm.testing.assert_allclose(out_topi, out_npy, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
