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
import tvm.testing
from tvm import relax
import json


# 0.13 BACKWARDS COMPATIBILITY TESTS
def test_vdevice():
    nodes = [
        {"type_key": ""},
        {
            "type_key": "relax.TensorStructInfo",
            "attrs": {
                "dtype": "float32",
                "ndim": "-1",
                "shape": "0",
                "span": "0",
            },
        },
    ]
    data = {
        "root": 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.13.0"},
        "b64ndarrays": [],
    }
    tsinfo = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tsinfo, relax.TensorStructInfo)
    assert not tsinfo.vdevice


if __name__ == "__main__":
    tvm.testing.main()
