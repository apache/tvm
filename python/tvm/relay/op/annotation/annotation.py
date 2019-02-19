"""Annotation operations."""
from __future__ import absolute_import as _abs
from . import _make
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
