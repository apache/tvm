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
"""Converts Relay IR subgraph to Android NNAPI source code."""
import tvm
from .converter import Converter


def convert_relayir_to_nnapi(func):
    """Converts a Relay IR Function to Android NNAPI C++ source code.

    Parameters
    ----------
    func: tvm.relay.Function
        The function to be converted to Android NNAPI.

    Returns
    -------
    code: str
        The resulting Android NNAPI code.

    Notes
    -----
    Certain function attributes should be configured:

    * func.attrs.NnapiClassName: (str) The name of the generated class wrapped around ANN model.
    * func.attrs.NnapiTargetVersion: (int) The targeting API level of Android.
    """
    assert isinstance(func, tvm.relay.Function)

    options = {
        "class": {
            "self": {
                "name": str(func.attrs.NnapiClassName),
            },
        },
        "target": {
            "api_level": int(func.attrs.NnapiTargetVersion),
        },
    }
    converter = Converter(options)
    return converter.convert(func)


tvm.register_func("relay.ext.android_nnapi.convert_relayir_to_nnapi", convert_relayir_to_nnapi)
