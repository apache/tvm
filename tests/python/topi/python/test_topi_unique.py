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
def test_unique(dev, target):
    def calc_numpy_unique(data, is_sorted=False):
        uniq, index, inverse, counts = np.unique(
            data, return_index=True, return_inverse=True, return_counts=True
        )
        num_uniq = np.array([len(uniq)]).astype("int32")
        if not is_sorted:
            order = np.argsort(index)
            reverse_order = np.argsort(order)
            uniq = uniq[order].astype(data.dtype)
            inverse = np.array([reverse_order[i] for i in inverse]).astype("int32")
            counts = counts[order].astype("int32")
        return [uniq.astype(data.dtype), inverse.astype("int32"), counts, num_uniq]

    def check_unique(data, is_sorted=False):
        # numpy reference
        np_unique, np_indices, np_counts, np_num_unique = calc_numpy_unique(data, is_sorted)
        num_unique = np_num_unique[0]

        implementations = {
            "generic": (
                lambda x, return_counts: topi.unique(x, is_sorted, return_counts),
                topi.generic.schedule_unique,
            ),
            "cuda": (
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
        tvm_indices = tvm.nd.array(np.zeros(data.shape).astype("int32"), device=dev)
        tvm_num_unique = tvm.nd.array(np.zeros([1]).astype("int32"), device=dev)

        # without counts
        with tvm.target.Target(target):
            te_input = tvm.te.placeholder(shape=data.shape, dtype=str(data.dtype))
            outs = fcompute(te_input, False)
            s = fschedule(outs)
            func = tvm.build(s, [te_input, *outs])
            func(tvm_data, tvm_unique, tvm_indices, tvm_num_unique)

        assert tvm_num_unique.numpy()[0] == np_num_unique
        np.testing.assert_allclose(tvm_unique.numpy()[:num_unique], np_unique, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(tvm_indices.numpy(), np_indices, atol=1e-5, rtol=1e-5)

        # with counts
        tvm_counts = tvm.nd.array(np.zeros(data.shape).astype("int32"), device=dev)
        with tvm.target.Target(target):
            te_input = tvm.te.placeholder(shape=data.shape, dtype=str(data.dtype))
            outs = fcompute(te_input, True)
            s = fschedule(outs)
            func = tvm.build(s, [te_input, *outs])
            func(tvm_data, tvm_unique, tvm_indices, tvm_num_unique, tvm_counts)

        np_unique, np_indices, _, np_num_unique = calc_numpy_unique(data, is_sorted)
        num_unique = np_num_unique[0]
        assert tvm_num_unique.numpy()[0] == np_num_unique
        np.testing.assert_allclose(tvm_unique.numpy()[:num_unique], np_unique, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(tvm_indices.numpy(), np_indices, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(tvm_counts.numpy()[:num_unique], np_counts, atol=1e-5, rtol=1e-5)

    for in_dtype in ["int32", "int64"]:
        for is_sorted in [True, False]:
            data = np.random.randint(0, 100, size=(1)).astype(in_dtype)
            check_unique(data, is_sorted)
            data = np.random.randint(0, 10, size=(10)).astype(in_dtype)
            check_unique(data, is_sorted)
            data = np.random.randint(0, 100, size=(10000)).astype(in_dtype)
            check_unique(data, is_sorted)


if __name__ == "__main__":
    test_unique(tvm.device("cpu"), tvm.target.Target("llvm"))
    test_unique(tvm.device("cuda"), tvm.target.Target("cuda"))
    test_unique(tvm.device("nvptx"), tvm.target.Target("nvptx"))
