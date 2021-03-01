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
import tvm.relay
from tvm.relay import op


def test_pass_profiler():
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    tvm.transform.enable_pass_profiling()

    mod = tvm.relay.transform.AnnotateSpans()(mod)
    mod = tvm.relay.transform.ToANormalForm()(mod)
    mod = tvm.relay.transform.InferType()(mod)

    profiles = tvm.transform.render_pass_profiles()
    assert "AnnotateSpans" in profiles
    assert "ToANormalForm" in profiles
    assert "InferType" in profiles

    tvm.transform.clear_pass_profiles()
    tvm.transform.disable_pass_profiling()
