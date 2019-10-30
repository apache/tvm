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

from tvm import relay
from tvm.expr import IntImm

def verify_dispatch_function(name, low, high, extra_arg, input_shape):
    end = 2048
    intervals = []
    buckets = []
    if name == "uniform":
        for i in range(end):
            if (i + 1) * extra_arg > end:
                break
            intervals.append((i * extra_arg, (i + 1) * extra_arg))
        for interval in intervals:
            left = max(low, interval[0])
            if interval[1] <= left:
                continue
            right = min(high, interval[1])
            buckets.append((left, right))
            if interval[1] >= high:
                break
        buckets.append((high, -1))

        func = relay.utils.uniform_dispatcher(low, high, extra_arg)
        out = func(input_shape)
        for iname, ishape in input_shape.items():
            for i, dim in enumerate(ishape):
                if not isinstance(dim, (int, IntImm)):
                    assert out[iname][i] == buckets, "%s vs %s" % (out[iname][i], buckets)
    else:
        for i in range(high):
            if pow(extra_arg, i + 1) > end:
                break
            intervals.append((pow(extra_arg, i), pow(extra_arg, i + 1)))
        for interval in intervals:
            left = max(low, interval[0])
            if interval[1] <= left:
                continue
            right = min(high, interval[1])
            buckets.append((left, right))
            if interval[1] >= high:
                break
        buckets.append((high, -1))

        func = relay.utils.exp_dispatcher(low, high, extra_arg)
        out = func(input_shape)
        for iname, ishape in input_shape.items():
            for i, dim in enumerate(ishape):
                if not isinstance(dim, (int, IntImm)):
                    assert out[iname][i] == buckets, "%s vs %s" % (out[iname][i], buckets)



def test_dispatch_function():
    verify_dispatch_function("uniform", 1, 256, 16, {"data": (relay.Any(), 3, 2, 2)})
    verify_dispatch_function("exp", 1, 512, 4, {"data": (relay.Any(), 3, 2, 2)})
    verify_dispatch_function("uniform", 224, 1024, 32, {"data": (1, 3, relay.Any(), relay.Any())})
    verify_dispatch_function("exp", 224, 1024, 2, {"data": (1, 3, relay.Any(), relay.Any())})

if __name__ == "__main__":
    test_dispatch_function()
