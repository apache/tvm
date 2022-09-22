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
"""Functions to print doc into text format"""

from typing import Callable, Optional

from tvm._ffi import register_object
from tvm.runtime import Object, ObjectPath

from . import _ffi_api
from .doc import ExprDoc, IdDoc
from .frame import Frame


@register_object("script.printer.VarTable")
class VarTable(Object):
    """
    Variable Table manages mapping from variable object to ExprDoc during
    the process of printing TVMScript.
    """

    def __init__(self):
        """
        Create an empty VarTable.
        """
        self.__init_handle_by_constructor__(_ffi_api.VarTable)  # type: ignore # pylint: disable=no-member

    def define(self, obj: Object, name_hint: str, object_path: ObjectPath, frame: Frame) -> IdDoc:
        """
        Define a variable by name.

        Parameters
        ----------
        obj : Object
            The variable object.
        name_hint : str
            The hint for variable name.
        object_path : ObjectPath
            The object path to be associated with the returned ExprDoc.
        frame : Frame
            Then frame that this variable is defined in.

        Returns
        -------
        doc : IdDoc
            The doc for this variable.
        """
        return _ffi_api.VarTableDefine(self, obj, name_hint, object_path, frame)  # type: ignore # pylint: disable=no-member

    def define_by_doc(self, obj: Object, doc_factory: Callable[[], ExprDoc], frame: Frame) -> None:
        """
        Define a variable by ExprDoc.

        Parameters
        ----------
        obj : Object
            The variable object.
        doc_factory : Callable[[], ExprDoc]
            The hint for variable name.
        frame : Frame
            Then frame that this variable is defined in.

        Returns
        -------
        None
        """
        _ffi_api.VarTableDefineByDoc(self, obj, doc_factory, frame)  # type: ignore # pylint: disable=no-member

    def get_var_doc(self, obj: Object, object_path: ObjectPath) -> Optional[ExprDoc]:
        """
        Get the doc for a variable.

        Parameters
        ----------
        obj : Object
            The variable object.
        object_path : ObjectPath
            The object path to be associated with the returned ExprDoc.

        Returns
        -------
        doc : ExprDoc
            The doc for this variable.
        """
        return _ffi_api.VarTableGetVarDoc(self, obj, object_path)  # type: ignore # pylint: disable=no-member

    def is_var_defined(self, obj: Object) -> bool:
        """
        Check whether a variable is defined.

        Parameters
        ----------
        obj : Object
            The variable object.

        Returns
        -------
        is_defined : bool
            Whether the variable is defined.
        """
        return _ffi_api.VarTableIsVarDefined(self, obj)  # type: ignore # pylint: disable=no-member

    def __contains__(self, obj: Object) -> bool:
        return self.is_var_defined(obj)
