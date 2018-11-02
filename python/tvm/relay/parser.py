from __future__ import absolute_import

def enabled():
    try:
        import tvm.relay._parser
        return True
    except ImportError:
        return False

def fromtext(data):
    """Parse a Relay program."""
    from tvm.relay import _parser    
    return _parser.fromtext(data)
