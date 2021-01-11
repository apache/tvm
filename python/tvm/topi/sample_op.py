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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Scatter operator"""
from ..tir import decl_buffer, ir_builder, Cast, AssertStmt, StringImm, Evaluate
from ..te import extern, hybrid


@hybrid.script
def _sample_op(sample_input):
    out = output_tensor((sample_input[0],), "int64")
    for i in range(sample_input[0]):
        out[i] = int64(1)
    return out

def sample_op(sample_input):
    return _sample_op(sample_input)
