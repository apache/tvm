# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The Relay IR namespace containing the IR definition and compiler."""
from __future__ import absolute_import
from .base import NodeBase, register_relay_node
from ..api import register_func

@register_relay_node
class InterpreterState(NodeBase):
    pass

# pylint: disable=unused-argument
def _debugger_init(expr, stack):
    import pdb
    pdb.set_trace()

# pylint: disable=unused-argument
@register_func("relay.debug")
def _debug(*args):
    _, _, _, ist = args
    print("Relay Debugger")
    print("  You can manipulate the expression under evaluation with the name `expr`.")
    print("  You can manipulate the call stack with the name `stack`.")
    print("--------------")
    print("--------------")
    _debugger_init(ist.current_expr, ist.stack)
