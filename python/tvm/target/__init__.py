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
"""Target description and codegen module.

TVM uses JSON-based target configuration. Targets can be constructed via:

- A config dictionary: ``Target({"kind": "cuda", "arch": "sm_80"})``
- A tag name: ``Target("nvidia/nvidia-a100")``
- A tag with overrides: ``Target({"tag": "nvidia/nvidia-a100", "l2_cache_size_bytes": 12345})``
- A kind name: ``Target("cuda")``

Use ``target.attrs["key"]`` to access target attributes such as
``"arch"``, ``"max_num_threads"``, ``"mcpu"``, ``"libs"``, etc.

Use :py:func:`tvm.target.list_tags` to list all available target tags,
and :py:func:`tvm.target.register_tag` to register new tags.
"""
from .target import Target, TargetKind
from .virtual_device import VirtualDevice
from .tag import list_tags, register_tag
from . import datatype
from . import codegen
from . import tag_registry  # noqa: F401  -- registers tags on import
