"""Hybrid Programming APIs of TVM Python Package.

This package maps a subset of python to HalideIR
so that users can write some computation patterns
have not been supported yet and get HalideIR easily.
"""

from .api import hybrid_script, parse
