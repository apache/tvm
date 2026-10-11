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

"""Workspace tensor utilities for TRN operator scheduling."""

from tvm.tirx import Var

largest_psum_per_bank = 512
max_psum_banks = 8


def check_workspace_tensor(tensor: Var, shape: tuple[int], scope: str):
    """Check if a workspace tensor is valid.

    Parameters
    ----------
    tensor : Var
        The workspace tensor to check
    shape : Tuple[int]
        The required shape
    scope : str
        The required scope

    Raises
    ------
    AssertionError :
        If the tensor is invalid
    """
    assert tensor.scope() == scope, f"workspace tensor must be a {scope} tensor"
    assert tensor.ty.layout is None, "workspace tensor must not have a layout"
    if scope == "trn.psum":
        # the number of psum banks used is inferred from the shape
        # only check p and f dims
        assert all(x >= y for x, y in zip(tensor.ty.shape[1:], shape)), (
            f"workspace tensor must have enough size, {tensor.ty.shape[1:]} cannot cover {shape}"
        )
    else:
        assert all(x >= y for x, y in zip(tensor.ty.shape, shape)), (
            f"workspace tensor must have enough size, {tensor.ty.shape} cannot cover {shape}"
        )
