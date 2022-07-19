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
from tvm import target
from tvm.runtime import ndarray as _nd
from tvm.runtime import Device as _Device

from . import _make
from .. import op as reg


def _make_virtual_device(device):
    if isinstance(device, _Device):
        return target.VirtualDevice(device)
    if isinstance(device, str):
        return target.VirtualDevice(_nd.device(device))
    if isinstance(device, target.VirtualDevice):
        return device
    raise ValueError("expecting a Device or device name, but received a %s" % (type(device)))


def on_device(body, device, constrain_result=False, constrain_body=True):
    """Annotates a body expression with device constraints. The constraint influences
    how the body is compiled, where the body is evaluated, and where the result of
    evaluation is stored.

    Note that the defaults for the constrain_body and constrain_result parameters should
    almost never need to be overridden by the user. These parameters are exposed here
    to help unit tests exercise the PlanDevices pass machinery.

    Parameters
    ----------
    body : tvm.relay.Expr
        The expression to be annotated.

    device : Union[:py:class:`Device`, str]
        The device to annotate with.

    constrain_result  : bool
        If false (the default), the result of the on_device is not constrained to be on device.

    constrain_body : bool
        If true (the default), the body of the on_device is constrained to be on device.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    return _make.OnDevice(body, _make_virtual_device(device), constrain_result, constrain_body)


def function_on_device(function, param_devices, result_device):
    """Annotates a Relay function with the device types on which its parameters and result should
    be stored.

    Parameters
    ----------
    function : tvm.relay.Function
        The function to be annotated.

    param_devices : Array[Union[:py:class:`Device`, str]]
        The devices for each parameter.

    result_device: Union[:py:class:`Device`, str]
        The device for the function result.

    Returns
    -------
    result : tvm.relay.Function
        The annotated function.
    """
    return _make.FunctionOnDevice(
        function,
        [_make_virtual_device(d) for d in param_devices],
        _make_virtual_device(result_device),
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
