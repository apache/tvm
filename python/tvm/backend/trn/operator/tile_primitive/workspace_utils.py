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

"""Workspace buffer utilities for TRN operator scheduling."""

from tvm.tirx import Buffer

largest_psum_per_bank = 512
max_psum_banks = 8


def check_workspace_buffer(buffer: Buffer, shape: tuple[int], scope: str):
    """Check if a workspace buffer is valid.

    Parameters
    ----------
    buffer : Buffer
        The workspace buffer to check
    shape : Tuple[int]
        The required shape
    scope : str
        The required scope

    Raises
    ------
    AssertionError :
        If the buffer is invalid
    """
    assert buffer.scope() == scope, f"workspace buffer must be a {scope} buffer"
    assert buffer.layout is None, "workspace buffer must not have a layout"
    if scope == "trn.psum":
        # the number of psum banks used is inferred from the shape
        # only check p and f dims
        assert all(x >= y for x, y in zip(buffer.shape[1:], shape)), (
            f"workspace buffer must have enough size, {buffer.shape[1:]} cannot cover {shape}"
        )
    else:
        assert all(x >= y for x, y in zip(buffer.shape, shape)), (
            f"workspace buffer must have enough size, {buffer.shape} cannot cover {shape}"
        )
