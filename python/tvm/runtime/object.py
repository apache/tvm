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
# pylint: disable=invalid-name, unused-import
"""Runtime Object API"""

from tvm.ffi.core import Object
import tvm.ffi.core
from . import _ffi_node_api


def __object_dir__(obj):
    class_names = dir(obj.__class__)
    fnames = _ffi_node_api.NodeListAttrNames(obj)
    size = fnames(-1)
    return sorted([fnames(i) for i in range(size)] + class_names)


tvm.ffi.core._set_class_object(Object)
# override the default repr function for tvm.ffi.core.Object
tvm.ffi.core.__object_repr__ = _ffi_node_api.AsRepr
tvm.ffi.core.__object_save_json__ = _ffi_node_api.SaveJSON
tvm.ffi.core.__object_load_json__ = _ffi_node_api.LoadJSON
tvm.ffi.core.__object_getattr__ = _ffi_node_api.NodeGetAttr
tvm.ffi.core.__object_dir__ = __object_dir__
