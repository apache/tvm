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

"""Document reflected FFI methods using their registered signature docstrings."""

from autodocsumm import CallableAttributeDocumenter
from sphinx.ext.autodoc import ClassDocumenter, MethodDocumenter
from tvm_ffi import Function


class FFIMethodDocumenter(MethodDocumenter):
    """Keep native methods out of callable-attribute ``__call__`` inspection."""

    objtype = "ffimethod"
    directivetype = "method"
    priority = CallableAttributeDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, Function) and isinstance(parent, ClassDocumenter)


def setup(app):
    app.add_autodocumenter(FFIMethodDocumenter)
    return {"parallel_read_safe": True}
