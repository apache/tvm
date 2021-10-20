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
"""Annotation operations."""
from tvm.runtime import ndarray as _nd
from tvm.runtime import Device as _Device

from . import _make
from .. import op as reg


def _device_to_int(device):
    if isinstance(device, _Device):
        return device.device_type
    if isinstance(device, str):
        return _nd.device(device).device_type
    raise ValueError("expecting a Device or device name, but received a %s" % (type(device)))


def on_device(data, device, is_fixed=False):
    """Annotates an expression with the device type on which its result should be stored.

    Parameters
    ----------
    data : tvm.relay.Expr
        The expression to be annotated.

    device : Union[:py:class:`Device`, str]
        The device to annotate with. Only the device's type is significant.

    is_fixed : bool
        If false (the default), a device_copy
        If true, the annotation does not imply a device_copy may be inserted to
        reconcile the device of the data argument with the device for the context of the
        annotated expression.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    return _make.on_device(data, _device_to_int(device), is_fixed)


def function_on_device(function, param_devices, result_device):
    """Annotates a Relay function with the device types on which its parameters and result should
    be stored.

    Parameters
    ----------
    function : tvm.relay.Function
        The function to be annotated.

    param_devices : Array[Union[:py:class:`Device`, str]]
        The devices for each parameter. Only the device types are significant.

    result_device: Union[:py:class:`Device`, str]
        The device for the function result. Only the device type is significant.

    Returns
    -------
    result : tvm.rleay.Function
        The annotated function.
    """
    return _make.function_on_device(
        function, [_device_to_int(d) for d in param_devices], _device_to_int(result_device)
    )


def stop_fusion(data):
    """Annotate an expression to prevent it being fused with following expressions.

    Parameters
    ----------
    data : tvm.relay.Expr
        The expression to be annotated.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    return _make.stop_fusion(data)


def checkpoint(data):
    """Annotate an expression to be a checkpoint for the checkpointing memory optimization.

    Parameters
    ----------
    data : tvm.relay.Expr
        The expression to be annotated.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    return _make.checkpoint(data)


reg.register_injective_schedule("annotation.checkpoint")


def compiler_begin(data, compiler):
    """Annotate an expression to indicate that it is the beginning of
    a regeion that will be handled by the given compiler.

    Parameters
    ----------
    data : tvm.relay.Expr
        The expression to be annotated.

    compiler : Str
        The compiler used to generate code of the annotated region.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    return _make.compiler_begin(data, compiler)


def compiler_end(data, compiler):
    """Annotate an expression to indicate that it is the end of a region that
    is handled by the provided compiler.

    Parameters
    ----------
    data : tvm.relay.Expr
        The expression to be annotated.

    compiler : Str
        The compiler used to generate code of the annotated region.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    return _make.compiler_end(data, compiler)
