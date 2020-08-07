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
"""
Responsible for converting function arguments into
a form that can be passed to a `PackedFunc`.
"""
import numpy as np
import tvm
from tvm import relay

def convert(a, ctx):
    """
    Converts a function input `a`
    (which may take constant defined in Relay, numpy arrays,
    or TVM NDArrays)
    into a form that can be passed to a TVM `PackedFunc`
    with the given context.
    """
    # convert(convert(a, tg), tg) = convert(a, tg)
    while True:
        if isinstance(a, int):
            a = np.array(a, dtype='int32')
        elif isinstance(a, np.ndarray):
            a = tvm.nd.array(a, ctx)
        elif isinstance(a, tvm.runtime.NDArray):
            return a
        elif isinstance(a, relay.Call):
            assert isinstance(a.op, relay.Constructor)
            a = (a.op, *a.args)
        elif isinstance(a, tuple):
            assert isinstance(a[0], relay.Constructor)
            a = relay.backend.interpreter.ConstructorValue(
                a[0].tag, [convert(arg, ctx) for arg in a[1:]], a[0])
        elif isinstance(a, relay.backend.interpreter.ConstructorValue):
            return a
        else:
            raise Exception(a, type(a))
