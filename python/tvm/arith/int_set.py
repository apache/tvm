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
"""Integer set."""
import tvm._ffi
from tvm.runtime import Object
from . import _ffi_api


class IntSet(Object):
    """Represent a set of integer in one dimension."""

    def is_nothing(self):
        """Whether the set represent nothing"""
        return _ffi_api.IntSetIsNothing(self)

    def is_everything(self):
        """Whether the set represent everything"""
        return _ffi_api.IntSetIsEverything(self)

    @staticmethod
    def vector(vec):
        """Construct an integer set that covers the vector expr

        Parameters
        ----------
        vec : PrimExpr
            The vector expression.

        Returns
        -------
        rset : IntSet
            The result set.
        """
        return _ffi_api.intset_vector(vec)

    @staticmethod
    def single_point(point):
        """Construct a point set.

        Parameters
        ----------
        point : PrimExpr
            The vector expression.

        Returns
        -------
        rset : IntSet
            The result set.
        """
        return _ffi_api.intset_single_point(point)


@tvm._ffi.register_object("arith.IntervalSet")
class IntervalSet(IntSet):
    """Represent set of continuous interval [min_value, max_value]

    Parameters
    ----------
    min_value : PrimExpr
        The minimum value in the interval.

    max_value : PrimExpr
        The maximum value in the interval.
    """

    def __init__(self, min_value, max_value):
        self.__init_handle_by_constructor__(_ffi_api.IntervalSet, min_value, max_value)
