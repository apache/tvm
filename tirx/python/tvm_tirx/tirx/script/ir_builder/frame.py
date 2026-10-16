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
"""IRBuilder for TIR"""

from tvm_ffi import register_object as _register_object

from tvm_tirx.script.ir_builder.frame import StmtFrame


@_register_object("script.ir_builder.tirx.FunctionFrame")
class FunctionFrame(StmtFrame):
    """Native function frame retaining signature and finalized results."""

    @property
    def params(self):
        """The native declared parameters, shared with the resumed body."""
        return self.args
