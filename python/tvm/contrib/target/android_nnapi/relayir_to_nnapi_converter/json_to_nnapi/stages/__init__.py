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
"""The steps and pipeline of Android NNAPI codegen
"""
from .symbolize import symbolize
from .declare_types import declare_types
from .declare_operands import declare_operands
from .declare_constants import declare_constants
from .declare_memories import declare_memories
from .initialize_operands import initialize_operands
from .declare_operations import declare_operations
from .declare_inputs_outputs import declare_inputs_outputs
from .declare_wrapper_class import declare_wrapper_class
from .set_execution_inputs_outputs import set_execution_inputs_outputs
from .finalize import finalize


STAGES = [
    # model creation
    symbolize,
    declare_types,
    declare_operands,
    declare_constants,
    declare_memories,
    initialize_operands,
    declare_operations,
    declare_inputs_outputs,
    # set execution io
    set_execution_inputs_outputs,
    # finalize
    declare_wrapper_class,
    finalize,
]
