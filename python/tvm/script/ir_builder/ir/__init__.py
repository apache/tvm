# isort: skip_file
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
"""Package tvm.script.ir_builder.ir"""

from tvm.ir import GenericConst, Range, StringImm, StringType
from tvm.script.parser.protocol_registry import constexpr

from ..base import IRBuilder, MISSING, annotation_value_, at_, with_at_group_, require_defined
from .parser_protocol import check_well_formed_, module_member_
from .frame import IRModuleFrame
from .ir import (
    decl_function,
    def_function,
    ir_module,
    module_attrs,
    module_get_attr,
    module_set_attr,
    module_global_infos,
    lookup_name,
    meta_var,
)
