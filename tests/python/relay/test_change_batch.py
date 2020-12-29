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
from tvm import te
from tvm import relay
from tvm.relay.testing import synthetic
from tvm.relay import transform


def test_change_batch_synthetic():
    net, params = synthetic.get_workload()
    new_net = transform.ChangeBatch({net["main"].params[0]: 0}, batch_size=123)(net)
    assert new_net["main"].checked_type.ret_type.shape[0] == 123


if __name__ == "__main__":
    test_change_batch_synthetic()
