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
"""tvm.contrib.msc.core.transform.transform"""

import tvm
from tvm.relax.transform import _ffi_api as relax_api
from tvm.relay.transform import _ffi_api as relay_api


def SetExprName(as_relax=True, entry_name="main") -> tvm.ir.transform.Pass:
    """Set name for the call and constant in IRModule.

    Parameters
    ----------
    as_relax: bool
        Whether set names for relax, otherwise for relay.
    entry_name: str
        The entry name


    Returns
    -------
    ret: tvm.ir.transform.Pass
    """

    if as_relax:
        return relax_api.SetRelaxExprName(entry_name)  # type: ignore
    return relay_api.SetRelayExprName(entry_name)  # type: ignore


def SetExprLayout(allow_missing=True, entry_name="main") -> tvm.ir.transform.Pass:
    """Set layout for the var and constant in IRModule.

    Parameters
    ----------
    allow_missing: bool
        Whether allow missing layouts.
    entry_name: str
        The entry name

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """

    return relax_api.SetExprLayout(allow_missing, entry_name)  # type: ignore
