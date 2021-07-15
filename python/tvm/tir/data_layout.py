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
"""Data layout."""
from typing import Union

import tvm._ffi

from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("tir.Layout")
class Layout(Object):
    """Layout is composed of upper cases, lower cases and numbers,
    where upper case indicates a primal axis and
    the corresponding lower case with factor size indicates the subordinate axis.
    For example, NCHW16c can describe a 5-D tensor of
    [batch_size, channel, height, width, channel_block].
    Here subordinate axis channel_block=16 is the factor size of the primal axis C (channel).

    See Also
    --------
    layout : Declare a layout
    """

    def __len__(self):
        return _ffi_api.LayoutNdim(self)  # type: ignore

    def __contains__(self, axis):
        return len(axis) == 1 and axis[0].isalpha() and axis[0] in self.name

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Layout index out of range")
        return _ffi_api.LayoutGetItem(self, index)  # type: ignore

    def index_of(self, axis):
        """Get the index of an axis

        Parameters
        ----------
        axis : str
            The axis name, need to be [a-z,A-Z]

        Returns
        -------
        index : int
            The index of the axis, -1 if not found.
        """
        return _ffi_api.LayoutIndexOf(self, axis)  # type: ignore

    def factor_of(self, axis):
        """Get the factor size of the subordinate axis.

        Parameters
        ----------
        axis : str
            The axis name, need to be [a-z,A-Z]

        Returns
        -------
        factor : int
            the size of the subordinate-axis of axis (if axis is a primal-axis),
            or the size of axis itself (if axis is a subordinate-axis).
            Return -1 if axis is not in the layout.
        """
        return _ffi_api.LayoutFactorOf(self, axis)  # type: ignore


@tvm._ffi.register_object("tir.BijectiveLayout")
class BijectiveLayout(Object):
    """Bijective mapping for two layouts (src-layout and dst-layout).
    It provides shape and index conversion between each other.

    Do not construct directly, use :any:`bijective_layout` instead.
    See the documentation of :any:`bijective_layout` for more details.

    Parameters
    ----------
    src_layout : str or Layout
        source layout.

    dst_layout : str or Layout
        destination layout.

    See Also
    --------
    bijective_layout : Declare a layout
    """

    def forward_index(self, index):
        """Given the indices of the src-layout, infer the dst index.

        Parameters
        ----------
        index: Array of Expr
            The indices in src-layout.

        Returns
        -------
        dst_index: Array of Expr
            The inferred indices in dst-layout.
        """
        return _ffi_api.BijectiveLayoutForwardIndex(self, index)  # type: ignore

    def backward_index(self, index):
        """Given the indices of the dst-layout, infer the src index.

        Parameters
        ----------
        index: Array of Expr
            The indices in dst-layout.

        Returns
        -------
        src_index: Array of Expr
            The inferred indices in src-layout.
        """
        return _ffi_api.BijectiveLayoutBackwardIndex(self, index)  # type: ignore

    def forward_shape(self, shape):
        """Given the shape of the src-layout, infer the dst shape.

        Parameters
        ----------
        shape: Array of Expr
            The shape in src-layout.

        Returns
        -------
        dst_shape: Array of Expr
            The inferred shape in dst-layout.
        """
        return _ffi_api.BijectiveLayoutForwardShape(self, shape)  # type: ignore

    def backward_shape(self, shape):
        """Given the shape of the dst-layout, infer the src shape.

        Parameters
        ----------
        shape: Array of Expr
            The shape in dst-layout.

        Returns
        -------
        src_shape: Array of Expr
            The inferred shape in src-layout.
        """
        return _ffi_api.BijectiveLayoutBackwardShape(self, shape)  # type: ignore


def layout(layout_str: str) -> Layout:
    """Create a layout node from a string.

    Parameters
    ----------
    layout_str : str
        A layout representation is composed of upper cases, lower cases and numbers,
        where upper case indicates a primal axis and
        the corresponding lower case with factor size indicates the subordinate axis.
        For example, NCHW16c can describe a 5-D tensor of
        [batch_size, channel, height, width, channel_block].
        Here subordinate axis channel_block=16 is the factor size of
        the primal axis C (channel).

    Returns
    -------
    layout : Layout
        The created layout
    """
    return _ffi_api.Layout(layout_str)  # type: ignore


def bijective_layout(
    src_layout: Union[str, Layout], dst_layout: Union[str, Layout]
) -> BijectiveLayout:
    """Create a bijective layout mapping.

    Parameters
    ----------
    src_layout : str or Layout
        source layout.

    dst_layout : str or Layout
        destination layout.

    Returns
    -------
    bijective_layout : BijectiveLayout
        The created bijective layout
    """
    if isinstance(src_layout, str):
        src_layout = layout(src_layout)
    if isinstance(dst_layout, str):
        dst_layout = layout(dst_layout)
    return _ffi_api.BijectiveLayout(src_layout, dst_layout)  # type: ignore
