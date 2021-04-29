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
"""Transform Relay IR patterns that's not suitable to lower to Android
NNAPI
"""

import tvm.relay
from .convert_scalar_to_tensor_for_broadcast_operators import (
    ConvertScalarToTensorForBroadcastOperators,
)


class FixIllegalPatternForNnapi:
    def __call__(self, func):
        assert isinstance(func, tvm.relay.Function)
        passes = [ConvertScalarToTensorForBroadcastOperators()]
        func = tvm.relay.transform.InferType()(tvm.IRModule({"main": func}))["main"]
        for p in passes:
            func = p(func)
            func = tvm.relay.transform.InferType()(tvm.IRModule({"main": func}))["main"]
        return func
