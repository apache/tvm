"""A parser for Relay's text format."""
from __future__ import absolute_import
from .. import register_func

def enabled():
    """Is the parser enabled/Can we import the parser?"""
    try:
        # pylint: disable=unused-variable
        from tvm.relay import _parser
        return True
    # pylint: disable=broad-except
    except Exception:
        return False

@register_func("relay.fromtext")
def fromtext(data, source_name=None):
    """Parse a Relay program."""
    from tvm.relay import _parser
    return _parser.fromtext(data, source_name)
