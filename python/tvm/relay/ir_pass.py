# pylint: disable=no-else-return,
# pylint: disable=unidiomatic-typecheck
"""The set of passes for Relay.

Exposes an interface for configuring the passes and scripting
them in Python.
"""
from . import _ir_pass

# Expose checking expression, should rename to infer_type.
# pylint: disable=invalid-name
check_expr = _ir_pass.check_expr

well_formed = _ir_pass.well_formed

check_kind = _ir_pass.check_kind

free_vars = _ir_pass.free_vars

free_type_vars = _ir_pass.free_type_vars
