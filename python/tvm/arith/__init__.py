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
"""Integer bound analysis, simplification and pattern detection."""

from .analyzer import Analyzer, ConstIntBound, Extension, ModularSet, ProofStrength
from .bound import deduce_bound
from .int_set import (
    IntervalSet,
    IntSet,
    PresburgerSet,
    estimate_region_lower_bound,
    estimate_region_strict_bound,
    estimate_region_upper_bound,
)
from .int_solver import solve_linear_equations, solve_linear_inequalities
from .iter_affine_map import (
    IterMapExpr,
    IterMark,
    IterSplitExpr,
    IterSumExpr,
    detect_iter_map,
    inverse_affine_iter_map,
    iter_map_simplify,
    normalize_iter_map_to_expr,
    normalize_to_iter_sum,
    subspace_divide,
)
from .pattern import detect_clip_bound, detect_common_subexpr, detect_linear_equation
