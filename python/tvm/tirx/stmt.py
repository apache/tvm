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
"""TIRx-specific statement nodes and buffer-region construction."""

from tvm.ir import Range, TensorRegion, Var

from . import _ffi_api


def BufferRegion(buffer: Var, region: list[Range]) -> TensorRegion:
    """Construct a buffer-backed tensor region with TIRX subscript semantics.

    Parameters
    ----------
    buffer : Var
        The source buffer.

    region : List[Range]
        The ranges, with one entry for each buffer dimension.
    """
    return _ffi_api.BufferRegion(buffer, region)
