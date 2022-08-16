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
"""
This file contains the entry point of TVMScript Unified Printer.
"""

from typing import Dict, Optional

from tvm.runtime.object_path import ObjectPath

from . import _ffi_api


def as_script(
    root_node,
    ir_name: str,
    ir_prefix: Dict[str, str],
    indent_spaces: int = 4,
    print_line_numbers: bool = False,
    num_context_lines: int = -1,
    path_to_underline: Optional[ObjectPath] = None,
) -> str:
    return _ffi_api.AsScript(
        root_node,
        ir_name,
        ir_prefix,
        indent_spaces,
        print_line_numbers,
        num_context_lines,
        path_to_underline,
    )
