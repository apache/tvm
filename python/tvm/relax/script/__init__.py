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
"""Relax-layer TVMScript pieces (parser, builder).

After the per-dialect TVMScript restructure, the Relax layer owns its own
``script/{parser,builder}`` subpackages. ``tvm.script.relax`` resolves to
this module via the dialect registry, so the public parser surface
(``function``, ``Tensor``, ``match_cast``, etc.) is re-exported here.
"""

# pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
from .parser import *
from .parser import dist
