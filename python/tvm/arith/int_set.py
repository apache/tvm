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


@tvm._ffi.register_object("arith.PresburgerSet")
class PresburgerSet(IntSet):
    """Represent of Presburger Set"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.PresburgerSet)


def estimate_region_lower_bound(region, var_dom, predicate):
    """Analyze the region with affine map, given the domain of variables and their predicate
    Some subregion may be discarded during the lower-bound analysis.

    Parameters
    ----------
    region : List[Range]
        The region to be analyzed.

    var_dom : Dict[Var, Range]
        The ranges of the variables

    predicate : PrimExpr
        The predicate for the affine map

    Returns
    ----------
    region_int_set : Optional[List[IntSet]]
        None if the detection fails, or an array of IntSets as the result of analysis
    """
    return _ffi_api.EstimateRegionLowerBound(region, var_dom, predicate)


def estimate_region_strict_bound(region, var_dom, predicate):
    """Analyze the region with affine map, given the domain of variables and their predicate
    The result should be strict, i.e. no region is discarded or relaxed.

    Parameters
    ----------
    region : List[Range]
        The region to be analyzed.

    var_dom : Dict[Var, Range]
        The ranges of the variables

    predicate : PrimExpr
        The predicate for the affine map

    Returns
    ----------
    region_int_set : Optional[List[IntSet]]
        None if the detection fails, or an array of IntSets as the result of analysis
    """
    return _ffi_api.EstimateRegionStrictBound(region, var_dom, predicate)


def estimate_region_upper_bound(region, var_dom, predicate):
    """Analyze the region with affine map, given the domain of variables and their predicate
    Relaxation of the region may be used in upper-bound analysis,
    i.e. some extra region may be added to the result.

    Parameters
    ----------
    region : List[Range]
        The region to be analyzed.

    var_dom : Dict[Var, Range]
        The ranges of the variables

    predicate : PrimExpr
        The predicate for the affine map

    Returns
    ----------
    region_int_set : List[IntSet]
        an array of IntSets as the result of analysis
    """
    return _ffi_api.EstimateRegionUpperBound(region, var_dom, predicate)


def pos_inf():
    """Returns the symbolic positive infinity

    Returns
    ----------
    pos_inf : Var
        A symbolic var that indicates positive infinity
    """
    return _ffi_api.PosInf()


def neg_inf():
    """Returns the symbolic positive infinity

    Returns
    ----------
    neg_inf : Var
        A symbolic var that indicates positive infinity
    """
    return _ffi_api.NegInf()


def union_lower_bound(sets):
    """Create a lower-bound of union set, where some of the segments may be dropped

    Parameters
    ----------
    sets : List[IntSet]
        The sets to be combined

    Returns
    ----------
    union_lower_bound : List[IntSet]
        An N-dimensional integer set, the lower bound of the union
    """
    return _ffi_api.UnionLowerBound(sets)
