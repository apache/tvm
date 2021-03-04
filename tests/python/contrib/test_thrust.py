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
import tvm
import tvm.testing
from tvm import te
from tvm.topi.cuda import stable_sort_by_key_thrust
from tvm.topi.cuda.scan import exclusive_scan, scan_thrust, schedule_scan
from tvm.contrib.thrust import can_use_thrust, can_use_rocthrust
import numpy as np


thrust_check_func = {"cuda": can_use_thrust, "rocm": can_use_rocthrust}


def test_stable_sort_by_key():
    size = 6
    keys = te.placeholder((size,), name="keys", dtype="int32")
    values = te.placeholder((size,), name="values", dtype="int32")

    keys_out, values_out = stable_sort_by_key_thrust(keys, values)

    for target in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.stable_sort_by_key"):
                print("skip because thrust is not enabled...")
                return

            ctx = tvm.context(target, 0)
            s = te.create_schedule([keys_out.op, values_out.op])
            f = tvm.build(s, [keys, values, keys_out, values_out], target)

            keys_np = np.array([1, 4, 2, 8, 2, 7], np.int32)
            values_np = np.random.randint(0, 10, size=(size,)).astype(np.int32)
            keys_np_out = np.zeros(keys_np.shape, np.int32)
            values_np_out = np.zeros(values_np.shape, np.int32)
            keys_in = tvm.nd.array(keys_np, ctx)
            values_in = tvm.nd.array(values_np, ctx)
            keys_out = tvm.nd.array(keys_np_out, ctx)
            values_out = tvm.nd.array(values_np_out, ctx)
            f(keys_in, values_in, keys_out, values_out)

            ref_keys_out = np.sort(keys_np)
            ref_values_out = np.array([values_np[i] for i in np.argsort(keys_np)])
            tvm.testing.assert_allclose(keys_out.asnumpy(), ref_keys_out, rtol=1e-5)
            tvm.testing.assert_allclose(values_out.asnumpy(), ref_values_out, rtol=1e-5)


def test_exclusive_scan():
    for target in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.sum_scan"):
                print("skip because thrust is not enabled...")
                return

            for ishape in [(10,), (10, 10), (10, 10, 10)]:
                values = te.placeholder(ishape, name="values", dtype="int32")

                scan, reduction = exclusive_scan(values, return_reduction=True)
                s = schedule_scan([scan, reduction])

                ctx = tvm.context(target, 0)
                f = tvm.build(s, [values, scan, reduction], target)

                values_np = np.random.randint(0, 10, size=ishape).astype(np.int32)
                values_np_out = np.zeros(values_np.shape, np.int32)

                if len(ishape) == 1:
                    reduction_shape = ()
                else:
                    reduction_shape = ishape[:-1]

                reduction_np_out = np.zeros(reduction_shape, np.int32)

                values_in = tvm.nd.array(values_np, ctx)
                values_out = tvm.nd.array(values_np_out, ctx)
                reduction_out = tvm.nd.array(reduction_np_out, ctx)
                f(values_in, values_out, reduction_out)

                ref_values_out = np.cumsum(values_np, axis=-1, dtype="int32") - values_np
                tvm.testing.assert_allclose(values_out.asnumpy(), ref_values_out, rtol=1e-5)
                ref_reduction_out = np.sum(values_np, axis=-1)
                tvm.testing.assert_allclose(reduction_out.asnumpy(), ref_reduction_out, rtol=1e-5)


def test_inclusive_scan():
    out_dtype = "int64"

    for target in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.sum_scan"):
                print("skip because thrust is not enabled...")
                return

            for ishape in [(10,), (10, 10)]:
                values = te.placeholder(ishape, name="values", dtype="int32")

                scan = scan_thrust(values, out_dtype, exclusive=False)
                s = tvm.te.create_schedule([scan.op])

                ctx = tvm.context(target, 0)
                f = tvm.build(s, [values, scan], target)

                values_np = np.random.randint(0, 10, size=ishape).astype(np.int32)
                values_np_out = np.zeros(values_np.shape, out_dtype)
                values_in = tvm.nd.array(values_np, ctx)
                values_out = tvm.nd.array(values_np_out, ctx)
                f(values_in, values_out)

                ref_values_out = np.cumsum(values_np, axis=-1, dtype=out_dtype)
                tvm.testing.assert_allclose(values_out.asnumpy(), ref_values_out, rtol=1e-5)


if __name__ == "__main__":
    test_stable_sort_by_key()
    test_exclusive_scan()
    test_inclusive_scan()
