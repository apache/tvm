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
from tvm.relay import testing


def verify_graph_dispatcher(mod, input_shape, dispatch_func):
    updated_mod = relay.transform.add_dispatch_func(mod, "main", input_shape, dispatch_func)
    buckets = dispatch_func(input_shape)

    num_buckets = 0
    for val in buckets.values():
        for buckets in val.values():
            if num_buckets == 0:
                num_buckets = len(buckets)
            else:
                num_buckets *= len(buckets)

    num_main_copies = 0
    for global_var in updated_mod.get_global_vars():
        if "main_copy_" in global_var.name_hint:
            num_main_copies += 1

    assert num_main_copies == num_buckets, "Expect %d main function copies but got %d" % (num_buckets, num_main_copies)

def test_graph_dispatcher():
    mod, _ = relay.testing.resnet.get_workload(num_layers=50, batch_size=relay.Any())
    verify_graph_dispatcher(mod, {"data": (relay.Any(), 3, 224, 224)}, relay.utils.uniform_dispatcher())
    mod, _ = relay.testing.densenet.get_workload(batch_size=1, image_shape=(3, relay.Any(), relay.Any()))
    verify_graph_dispatcher(mod, {"data": (1, 3, relay.Any(), relay.Any())}, relay.utils.exp_dispatcher())
    mod, _ = relay.testing.dcgan.get_workload(batch_size=2, random_len=relay.Any())
    verify_graph_dispatcher(mod, {"data": (2, relay.Any())}, relay.utils.uniform_dispatcher())
    mod, _ = relay.testing.dqn.get_workload(batch_size=3, image_shape=(4, relay.Any(), relay.Any()))
    verify_graph_dispatcher(mod, {"data": (3, 4, relay.Any(), relay.Any())}, relay.utils.uniform_dispatcher())

if __name__ == "__main__":
    test_graph_dispatcher()
