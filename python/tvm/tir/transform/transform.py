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
"""Wrapping existing transformations."""
# pylint: disable=invalid-name

from . import _ffi_api


def CombineContextCall():
    """Combine context calls in the host function.

    Returns
    -------
    fpass : tvm.ir.transform.Pass
        The result pass
    """
    return _ffi_api.CombineContextCall()


def LowerIntrin():
    """Lower target specific intrinsic calls.

    Returns
    -------
    fpass : tvm.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerIntrin()


def LowerDeviceStorageAccessInfo():
    """Lower attached storage access information on device.

    Returns
    -------
    fpass : tvm.ir.transform.Pass
        The result pass

    Note
    ----
    Run this pass after all storage access analysis finish.
    """
    return _ffi_api.LowerDeviceStorageAccessInfo()


def LowerWarpMemory():
    """Lower warp memory access to low-level device related function calls.

    Returns
    -------
    fpass : tvm.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerWarpMemory()


def NarrowDataType():
    """Narrow down PrimExpr datatype in stmt to target_bits.

    Returns
    -------
    fpass : tvm.ir.transform.Pass
        The result pass

    Note
    ----
    Run this pass after StorageFlatten.
    """
    return _ffi_api.NarrowDataType()
