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

import tvm.relay.testing
from mxnet.gluon.model_zoo.vision import get_model

from tvm import relay

#mod, params = relay.testing.resnet.get_workload(num_layers=50, batch_size=1, dtype="float32")
block = get_model('resnet50_v1', pretrained=True)
input_shape = {"data": (relay.Any(), 3, 224, 224)}
mod, params = relay.frontend.from_mxnet(block, shape=input_shape, dtype="float32")
updated_mod = relay.transform.dispatch_global_func(mod, "main", input_shape, relay.utils.exp_dispatcher())

