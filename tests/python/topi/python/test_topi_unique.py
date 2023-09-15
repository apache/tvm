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

in_dtype = tvm.testing.parameter("int32", "int64")
is_sorted = tvm.testing.parameter(True, False, ids=["sorted", "unsorted"])
with_indices = tvm.testing.parameter(True, False, ids=["with_indices", "no_indices"])
with_inverse_indices = tvm.testing.parameter(
    True, False, ids=["with_inverse_indices", "no_inverse_indices"]
)
with_counts = tvm.testing.parameter(True, False, ids=["with_counts", "no_counts"])
arr_size, maxval = tvm.testing.parameters((1, 100), (10, 10), (10000, 100))


@tvm.testing.parametrize_targets
def test_unique(
    dev,
    target,
    in_dtype,
    is_sorted,
    with_indices,
    with_inverse_indices,
    with_counts,
    arr_size,
    maxval,
):
    def calc_numpy_unique(data, is_sorted=False):
        uniq, index, inverse, counts = np.unique(
            data, return_index=True, return_inverse=True, return_counts=True
        )
        num_uniq = np.array([len(uniq)]).astype("int32")
        if not is_sorted:
            order = np.argsort(index)
            index = np.sort(index)
            reverse_order = np.argsort(order)
            uniq = uniq[order].astype(data.dtype)
            inverse = np.array([reverse_order[i] for i in inverse])
            counts = counts[order]
        return [
            uniq.astype(data.dtype),
            num_uniq,
            index.astype("int32"),
            inverse.astype("int32"),
            counts.astype("int32"),
        ]

    data = np.random.randint(0, maxval, size=(arr_size)).astype(in_dtype)

    # numpy reference
    np_unique, np_num_unique, np_indices, np_inverse_indices, np_counts = calc_numpy_unique(
        data, is_sorted
    )
    num_unique = np_num_unique[0]

    implementations = {
        "generic": (
            lambda x, return_indices, return_inverse_indices, return_counts: topi.unique(
                x, is_sorted, return_indices, return_inverse_indices, return_counts
            ),
            topi.generic.schedule_unique,
        ),
        "gpu": (
            lambda x, return_counts: topi.cuda.unique(x, is_sorted, return_counts),
            topi.cuda.schedule_scan,
        ),
        "nvptx": (
            lambda x, return_counts: topi.cuda.unique(x, is_sorted, return_counts),
            topi.cuda.schedule_scan,
        ),
    }
    fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)
    tvm_data = tvm.nd.array(data, device=dev)
    tvm_unique = tvm.nd.array(np.zeros(data.shape).astype(data.dtype), device=dev)
    tvm_num_unique = tvm.nd.array(np.zeros([1]).astype("int32"), device=dev)

    with tvm.target.Target(target):
        te_input = tvm.te.placeholder(shape=data.shape, dtype=str(data.dtype))
        outs = fcompute(te_input, with_indices, with_inverse_indices, with_counts)
        s = fschedule(outs)
        func = tvm.build(s, [te_input, *outs])
        tvm_args = [tvm_data, tvm_unique, tvm_num_unique]
        if with_indices:
            tvm_indices = tvm.nd.array(np.zeros(data.shape).astype("int32"), device=dev)
            tvm_args.append(tvm_indices)
        if with_inverse_indices:
            tvm_inverse_indices = tvm.nd.array(np.zeros(data.shape).astype("int32"), device=dev)
            tvm_args.append(tvm_inverse_indices)
        if with_counts:
            tvm_counts = tvm.nd.array(np.zeros(data.shape).astype("int32"), device=dev)
            tvm_args.append(tvm_counts)
        func(*tvm_args)

    assert tvm_num_unique.numpy()[0] == np_num_unique

    np.testing.assert_allclose(tvm_unique.numpy()[:num_unique], np_unique, atol=1e-5, rtol=1e-5)
    if with_indices:
        np.testing.assert_allclose(
            tvm_indices.numpy()[:num_unique], np_indices, atol=1e-5, rtol=1e-5
        )

    if with_inverse_indices:
        np.testing.assert_allclose(
            tvm_inverse_indices.numpy(), np_inverse_indices, atol=1e-5, rtol=1e-5
        )

    if with_counts:
        np.testing.assert_allclose(tvm_counts.numpy()[:num_unique], np_counts, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
