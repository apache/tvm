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
"""Relax IR analysis. """

from .analysis import (
    BaseCheckResult,
    all_global_vars,
    all_vars,
    bound_vars,
    computable_at_compile_time,
    contains_impure_call,
    definable_tir_vars_in_struct_info,
    defined_symbolic_vars,
    derive_call_ret_struct_info,
    detect_recursion,
    erase_to_well_defined,
    free_symbolic_vars,
    free_vars,
    get_static_type,
    get_var2val,
    has_reshape_pattern,
    name_to_binding,
    post_order_visit,
    remove_all_unused,
    struct_info_base_check,
    struct_info_lca,
    suggest_layout_transforms,
    tir_vars_in_struct_info,
    udchain,
    well_formed,
)
from .estimate_memory_usage import estimate_memory_usage
