from __future__ import absolute_import

def enabled():
    """Is the parser enabled/Can we import the parser?"""
    try:
        # pylint: disable=unused-variable
        import tvm.relay._parser
        return True
    # pylint: disable=broad-except
    except Exception:
        return False

def fromtext(data):
    """Parse a Relay program."""
    from tvm.relay import _parser
    return _parser.fromtext(data)
