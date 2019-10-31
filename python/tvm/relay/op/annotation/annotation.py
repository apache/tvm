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
from __future__ import absolute_import as _abs
from . import _make
from ..op import register_schedule, schedule_injective
from .... import nd as _nd
from .... import TVMContext as _TVMContext

def on_device(data, device):
    """Annotate an expression with a certain device type.

    Parameters
    ----------
    data : tvm.relay.Expr
        The expression to be annotated.

    device : Union[:py:class:`TVMContext`, str]
        The device type to annotate.

    Returns
    -------
    result : tvm.relay.Expr
        The annotated expression.
    """
    if isinstance(device, _TVMContext):
        device = device.device_type
    elif isinstance(device, str):
        device = _nd.context(device).device_type
    else:
        raise ValueError("device is expected to be the type of TVMContext or "
                         "str, but received %s" % (type(device)))
    return _make.on_device(data, device)


def stop_fusion(data):
    """Annotate an expression to prevent it being fused with previous expressions.

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

register_schedule("annotation.checkpoint", schedule_injective)
