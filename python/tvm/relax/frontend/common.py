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
"""Commons for Relax frontend."""
from typing import Dict, List, Optional

import tvm


class ImporterOutput:
    """The data structure representing the result of frontend imports.

    Attributes
    ----------
    mod : tvm.IRModule
        The IRModule imported from frontend.

    params : Optional[Dict[str, List[tvm.nd.NDArray]]]
        The weights of the imported model, when the weights of the model are
        requested to be kept as parameters of functions in the IRModule. (e.g.,
        when the `keep_params_as_input` flag of `frontend.torch.from_fx` is set to
        True.)
        - `params` is defined to be None when not requested.
        - The keys of `params` are the names of the Relax functions in the IRModule.
        - Each weight tensor is in the form of TVM NDArray on device CPU.
        - The order of the returned weights is in accordance with the order of
        the kept Relax function input variables.
    """

    mod: tvm.IRModule
    params: Optional[Dict[str, List[tvm.nd.NDArray]]]

    def __init__(self, mod: tvm.IRModule, params: Optional[Dict[str, List[tvm.nd.NDArray]]]):
        self.mod = mod
        self.params = params
