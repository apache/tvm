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
"""
IRDocsifier is the top-level interface in the process of transforming
IR graph into Doc tree, during printing IR graph as TVMScript code.
"""

import atexit
from contextlib import ExitStack, contextmanager
from typing import Callable, Dict, Generator, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar

from tvm._ffi import get_object_type_index, register_object
from tvm.runtime import Object, ObjectPath

from . import _ffi_api
from .doc import Doc
from .frame import Frame
from .var_table import VarTable

_REGISTERED_TYPES: Set[Tuple[str, int]] = set()  # {(dispatch_token, type_index)}


def _cleanup_dispatch_function():
    for dispatch_token, type_index in _REGISTERED_TYPES:
        _ffi_api.IRDocsifierRemoveDispatch(dispatch_token, type_index)  # type: ignore # pylint: disable=no-member


_CLEANUP_REGISTERED = False


def _ensure_cleanup_function_registered():
    """
    Add a cleanup function to be called on interpreter termination,
    to remove all dispatch functions registered on the Python side.

    Without cleaning up those dispatch functions, program will segfault
    on termination. It's because dispatch functions are referenced from the
    static memory of libtvm, thus they will be cleaned up at the very end,
    making calls to Py_DecRef after Python interpreter terminates.
    """
    global _CLEANUP_REGISTERED  # pylint: disable=global-statement

    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup_dispatch_function)
        _CLEANUP_REGISTERED = True


@register_object("script.printer.RootNodeContainer")
class RootNodeContainer(Object):
    """
    A wrapper object to provide injection point for printer of each IR.

    This class shouldn't be used directly. `IRDocsifier.set_root_dispatch`
    should be used instead.
    """

    root_node: Object

    def __init__(self, root_node: Object):
        self.__init_handle_by_constructor__(_ffi_api.RootNodeContainer, root_node)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.IRDocsifier")
class IRDocsifier(Object):
    """
    IRDocsifier is the top-level interface in the IR->Doc process.

    It provides methods to convert IR node object to Doc, operate on Frame
    objects and change dispatch tokens.
    """

    ir_prefix: Mapping[str, str]
    vars: VarTable
    frames: Sequence[Frame]
    dispatch_tokens: Sequence[str]

    def __init__(self, ir_prefix: Dict[str, str]):
        """
        Create a new IRDocsifier.

        Parameters
        ----------
        ir_prefix : Dict[str, str]
            The ir prefix to use. Key is the IR dispatch token and
            value is the name of identifier for this IR's namespace in TVMScript.
        """
        self.__init_handle_by_constructor__(_ffi_api.IRDocsifier, ir_prefix)  # type: ignore # pylint: disable=no-member

    _TObject = TypeVar("_TObject", bound=Object)

    @classmethod
    def set_dispatch(
        cls,
        node_type: Type[_TObject],
        dispatch_function: Callable[[_TObject, ObjectPath, "IRDocsifier"], Doc],
        dispatch_token: str = "",
    ) -> None:
        """
        Set the dispatch function to transform a particular IR node type to Doc

        Parameters
        ----------
        node_type : Type[_TObject]
            The type of object to dispatch on.
        dispatch_function : Callable[[_TObject, ObjectPath, "IRDocsifier"], Doc]
            The dispatch function. It's called to transform IR node object to Doc.
        dispatch_token : str
            Function will only be called when this dispatch_token is the same as the one
            on the top of IRDocsifier's dispatch_tokens stack. An empty dispatch token
            means registering as default dispatch function, which will be called when
            there is no dispatch function registered with the current dispatch token.
        """
        type_index = get_object_type_index(node_type)
        if type_index is None:
            raise TypeError(f"{type(node_type)} is not a registered TVM object type.")

        _ensure_cleanup_function_registered()
        _ffi_api.IRDocsifierSetDispatch(  # type: ignore # pylint: disable=no-member
            dispatch_token, type_index, dispatch_function
        )
        _REGISTERED_TYPES.add((dispatch_token, type_index))

    @classmethod
    def set_root_dispatch(
        cls, dispatch_token: str, root_dispatch_function: Callable[[Object, "IRDocsifier"], Doc]
    ) -> None:
        """
        Set the root dispatch function for an IR.

        The root dispatch function will be called with the root node of an IR graph
        that's being transformed to Doc. This provides an injection point for
        each IR's printer implemention to add specialized logic, for example,
        pushing a special Frame to the IRDocsifier before doing actual IR->Doc
        transformation.

        The simplest root dispatch function is
        ```
        def f(obj, ir_docsifier)
            return ir_docsifier.as_doc(obj, ObjectPath.root())
        ```

        Parameters
        ----------
        root_dispatch_function : Callable[[_TObject, "IRDocsifier"], Doc]
            The root dispatch function. It's called with the root node to be printed.
        dispatch_token : str
            The dispatch token of the IR that root_dispatch_funnction applies to.
        """

        def dispatch_function(obj: RootNodeContainer, _, ir_docsifier):
            return root_dispatch_function(obj.root_node, ir_docsifier)

        cls.set_dispatch(RootNodeContainer, dispatch_function, dispatch_token)

    def as_doc(self, obj: Object, object_path: ObjectPath) -> Doc:
        """
        Transform the input object into Doc.

        Parameters
        ----------
        obj : Object
            The IR node object.
        object_path : ObjectPath
            The object path of this object. It's used for locating diagnostic message.

        Returns
        -------
        doc : Doc
            The doc for this object.
        """
        return _ffi_api.IRDocsifierAsDoc(self, obj, object_path)  # type: ignore # pylint: disable=no-member

    def get_frame(self, frame_type: Type[Frame]) -> Optional[Frame]:
        """
        Get the top frame with type `frame_type`.

        Parameters
        ----------
        frame_type : Type[Frame]
            The target frame type.

        Returns
        -------
        frame : Optional[Frame]
            The frame if found, otherwise None.
        """
        for i in range(len(self.frames) - 1, -1, -1):
            if isinstance(self.frames[i], frame_type):
                return self.frames[i]
        return None

    @contextmanager
    def dispatch_token(self, token: str):
        """
        Push a new dispatch token to the stack.

        Parameters
        ----------
        token : str
            The token to push.

        Returns
        -------
        A context manager that pops this dispatch token when exits.
        """
        with ExitStack() as stack:
            _ffi_api.IRDocsifierPushDispatchToken(self, token)  # type: ignore # pylint: disable=no-member
            stack.callback(_ffi_api.IRDocsifierPopDispatchToken, self)  # type: ignore # pylint: disable=no-member
            yield

    _TFrame = TypeVar("_TFrame", bound=Frame)

    @contextmanager
    def frame(self, frame: _TFrame) -> Generator[_TFrame, None, None]:
        """
        Push a new frame to the stack.

        Parameters
        ----------
        frame : Frame
            The frame to push.

        Returns
        -------
        A context manager that pops this frame when exits.
        """
        with ExitStack() as stack:
            stack.enter_context(frame)
            _ffi_api.IRDocsifierPushFrame(self, frame)  # type: ignore # pylint: disable=no-member
            stack.callback(_ffi_api.IRDocsifierPopFrame, self)  # type: ignore # pylint: disable=no-member
            yield frame
