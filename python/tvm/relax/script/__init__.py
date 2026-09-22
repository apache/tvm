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
"""Public canonical TVMScript dialect namespace."""
import importlib as _importlib


def __getattr__(name):
    if name in ("builder", "tile"):
        return _importlib.import_module(__name__ + "." + name)
    if name.startswith("_") and name != "__all__":
        raise AttributeError(name)
    from tvm.script import parser as _parser
    _parser._initialize()
    if name in globals():
        return globals()[name]
    builder = _importlib.import_module(__name__ + ".builder")
    return getattr(builder, name)
