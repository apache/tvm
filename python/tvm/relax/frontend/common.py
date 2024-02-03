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
# pylint: disable=invalid-name
"""Commons for Relax frontend."""
from typing import Dict, List, Tuple

import tvm


def detach_params(mod: tvm.IRModule) -> Tuple[tvm.IRModule, Dict[str, List[tvm.nd.NDArray]]]:
    """Detach the attribute "params" in the functions of the input IRModule as
    separate dictionary of params.

    Parameters
    ----------
    mod : tvm.IRModule
        The IRModule whose functions' "param" attribute is going to be detached.

    Returns
    -------
    detached_mod : tvm.IRModule
        The IRModule after the detachment.

    params_dict : Dict[str, List[tvm.nd.NDArray]]
        The detached params. The dict keys corresponds to the names of the
        functions in the input IRModule that have attribute "params".
    """
    detached_mod = tvm.IRModule()
    params_dict = dict()
    for gv, func in mod.functions_items():
        if func.attrs is not None and "params" in func.attrs:
            params = list(func.attrs["params"])
            if not all([isinstance(param, tvm.nd.NDArray) for param in params]):
                raise ValueError(
                    'The value "params" attribute is expected to be a list of NDArray.'
                )
            params_dict[gv.name_hint] = params
            detached_mod[gv] = func.without_attr("params")
        else:
            detached_mod[gv] = func
    return detached_mod, params_dict
