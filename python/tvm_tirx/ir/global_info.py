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
"""Global Info."""

import tvm_ffi


@tvm_ffi.register_object("ir.GlobalInfo")
class GlobalInfo(tvm_ffi.Object):
    """Base class for module-level information referenced by the IR.

    Concrete subclasses are stored in ``IRModule.global_infos``, a map from names
    to arrays of global information objects.
    """

    def __eq__(self, other):
        """Compare two global info objects for structural equivalence."""
        return tvm_ffi.structural_equal(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Overload with structural equality."""
        return super().__eq__(other)
