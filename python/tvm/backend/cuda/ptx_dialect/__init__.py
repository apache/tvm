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
"""Table-driven PTX dialect prototype (``T.ptxd``).

One table (:mod:`.table`), one generic engine (:mod:`.engine`), thin
generators (:mod:`.gen_helpers``/``gen_stubs`, :mod:`.gen_coverage`). Importing this package
registers every table entry as a TVM Op with a generic codegen; the
``T.ptxd`` namespace itself is installed by the CUDA backend's
``register_backend()`` via ``script_namespaces()``.
"""

from .engine import PTXDNamespace, register_table
from .table import TABLE

register_table(TABLE)

__all__ = ["TABLE", "PTXDNamespace", "register_table"]
