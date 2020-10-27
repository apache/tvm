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
# pylint: disable=invalid-name, unused-argument
"""CoreML codegen supported operators."""
import tvm.ir
from tvm.contrib.target.coreml import _convert_map
from ...expr import Constant


def _register_coreml_op(op_name):
    """Register a function to check the given operator is supported by Core ML.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    """

    def _check_supported(expr):
        attrs, args = expr.attrs, expr.args
        if op_name == "nn.conv2d":
            if not isinstance(args[1], Constant):
                return False
            if attrs["kernel_layout"] not in ["HWIO", "OIHW"]:
                return False
        return True

    tvm.ir.register_op_attr(op_name, "target.coremlcompiler", _check_supported)


for op in _convert_map:
    _register_coreml_op(op)
