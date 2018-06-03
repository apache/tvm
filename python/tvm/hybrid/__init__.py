"""Hybrid Programming APIs of TVM Python Package.

This package maps a subset of python to HalideIR so that:
1. Users can write some preliminary versions of the computation patterns
have not been supported yet and verify it across the real execution and
python semantic emulation.
2. Developers can build HalideIR by writing Python code.
"""

from .api import script, parse
