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
"""Utility backend functions."""
from enum import Enum


class CallType(Enum):
    Packed = 0
    CPacked = 1
    Unpacked = 2


def _is_valid_modname(mod_name):
    """Determine if mod_name is a valid string to use inside function names"""
    if mod_name:
        try:
            mod_name.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    return True


def mangle_module_name(mod_name):
    if not _is_valid_modname(mod_name):
        raise ValueError(mod_name + " contains invalid characters")
    if mod_name:
        return "tvmgen_" + mod_name
    return "tvmgen"
