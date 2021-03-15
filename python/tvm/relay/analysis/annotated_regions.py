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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, unused-import
"""Regions used in Relay."""

from ...runtime import Object
from . import _ffi_api


class AnnotatedRegionSet(Object):
    """Class to represent a relay expression split into regions."""

    def __init__(self, expr, region_begin_op, region_end_op):
        """Construct regions from an expression.

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression from which to construct the regions.
        region_begin_op : tvm.ir.Op
            The region begin annotation.
        region_end_op : tvm.ir.Op
            The region end annotation.

        """
        self.__init_handle_by_constructor__(
            _ffi_api.AnnotatedRegionSet, expr, region_begin_op, region_end_op
        )

    def __len__(self):
        return len(self.regions)

    def get_region(self, expr):
        """Get the region an expression belongs to.

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression.

        Returns
        -------
        region
            The region containing the expression.
            None if not found.
        """
        return _ffi_api.GetRegion(self, expr)
