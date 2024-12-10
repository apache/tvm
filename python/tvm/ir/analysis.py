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

# pylint: disable=unused-import

"""Common analysis across all IR variants."""

from typing import Dict, List

import tvm
from . import _ffi_analysis_api as _ffi


def collect_call_map(
    module: "tvm.ir.IRModule",
) -> Dict["tvm.ir.GlobalVar", List["tvm.ir.GlobalVar"]]:
    """Collect the call map of a module

    Parameters
    ----------
    module: tvm.ir.IRModule
        The module to inspect

    Returns
    -------
    call_map: Dict[tvm.ir.GlobalVar, List[tvm.ir.GlobalVar]]
        A map from functions to the subroutines they call.

    """
    return _ffi.CollectCallMap(module)
