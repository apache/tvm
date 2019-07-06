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
import tvm.relay.module as _module
import tvm.relay.transform as _transform

def test_eta_expand_basic():
    x = relay.var('x', 'int32')
    orig = relay.Function([x], x)
    mod = _module.Module.from_expr(orig)
    seq = _transform.Sequential([_transform.EtaExpand()])
    with _transform.PassContext(opt_level=3):
        mod = seq(mod)

    got = mod["main"]

    y = relay.var('y', 'int32')
    expected = relay.Function([y], orig(y))
    gv = relay.GlobalVar("gv")
    mod[gv] = expected
    mod = _transform.InferType()(mod)
    expected = mod["gv"]
    assert(relay.analysis.alpha_equal(got, expected))

if __name__ == "__main__":
    test_eta_expand_basic()
