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
"""S-TIR function construction over the shared TIRx builder operations."""

from tvm.script.ir_builder import base as _base

# pylint: disable=wildcard-import,unused-wildcard-import
from tvm.tirx.script.builder import *  # noqa: F403
from tvm.tirx.script.builder import _ffi_api, _native


def function_(*, private=False, persistent=False, decl=False, span=None):
    """Enter an S-TIR declaration or definition frame.

    Parameters
    ----------
    private : bool
        Omit the global symbol when true.
    persistent : bool
        Mark the function as a persistent kernel.
    decl : bool
        Create a declaration frame for module signature collection.
    span : Span or source location, optional
        Source location of the function.
    """
    frame = (
        _ffi_api.DeclFunction(private, True, persistent)
        if decl
        else _native.prim_func(private=private, s_tir=True, persistent=persistent)
    )
    return _base.at_(span, frame)
