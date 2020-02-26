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
"""namespace of IR node builder make function

This namespace is used for developers. While you do not see any declarations.
The functions are automatically exported from C++ side via PackedFunc.

Each api is a PackedFunc that can be called in a positional argument manner.
You can use make function to build the IR node.
"""
import tvm._ffi
import tvm.ir
from tvm.ir import make_node as node
from tvm.tir import Call


def make_by_min_extent(min_value, extent):
    """Construct a Range by min and extent.

    This constructs a range in [min_value, min_value + extent)

    Parameters
    ----------
    min_value : PrimExpr
        The minimum value of the range.

    extent : PrimExpr
        The extent of the range.

    Returns
    -------
    rng : Range
        The constructed range.
    """
    return tvm.ir.Range.make_by_min_extent(min_value, extent)

tvm._ffi._init_api("tvm.make")
