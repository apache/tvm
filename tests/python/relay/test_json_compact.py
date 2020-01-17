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
from tvm import relay
import json

def test_type_var():
    # type var in 0.6
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.TypeVar",
         "attrs": {"kind": "0", "span": "0", "var": "2"}},
        {"type_key": "Variable",
         "attrs": {"dtype": "int32", "name": "in0"}},
        ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.load_json(json.dumps(data))
    assert isinstance(tvar, relay.TypeVar)
    assert tvar.name_hint == "in0"
    nodes[1]["type_key"] = "relay.GlobalTypeVar"
    tvar = tvm.load_json(json.dumps(data))
    assert isinstance(tvar, relay.GlobalTypeVar)
    assert tvar.name_hint == "in0"


if __name__ == "__main__":
    test_type_var()
