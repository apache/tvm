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

"""Phase 1 helpers for the XNNPACK Relax backend."""

from tvm.ir import IRModule


def partition_for_xnnpack(mod: IRModule) -> IRModule:
    """Return ``mod`` unchanged until XNNPACK operator support is implemented.

    Phase 1 only installs an opt-in runtime/codegen skeleton. It intentionally registers no
    supported operator patterns, so this helper must not mark any Relax subgraph for XNNPACK.
    """

    return mod
