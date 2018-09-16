# pylint: disable=no-else-return,
# pylint: disable=unidiomatic-typecheck
"""The set of passes for Relay.

Exposes an interface for configuring the passes and scripting
them in Python.
"""
from . import _ir_pass

# Expose checking expression, should rename to infer_type.
check_expr = _ir_pass.check_expr
