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
from tvm.relay.testing import resnet
from tvm.relay.analysis import count_layers


def test_layer_count():
    def verify(num_layers):
        # Load a resnet with a known number of layers.
        mod, _ = resnet.get_workload(num_layers=num_layers)
        # Count the number of conv and dense layers.
        count = count_layers(mod, valid_ops=["nn.conv2d", "nn.dense"])
        assert count == num_layers

    verify(18)
    verify(50)


if __name__ == "__main__":
    test_layer_count()
