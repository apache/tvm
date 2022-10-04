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
"""TVMScript for TIR"""

# Type system
from .ty import void, boolean, handle, Ptr, Tuple, Buffer
from .ty import bool  # pylint: disable=redefined-builtin

from .prim_func import prim_func

# add all floating point and integer datatypes to the module
for _dtype in ["float", "uint", "int"]:
    for _size in ["8", "16", "32", "64"]:
        for _lanes in ["", "x4", "x8", "x16", "x32"]:
            from . import ty

            _name = _dtype + _size + _lanes
            globals()[_name] = getattr(ty, _name)
