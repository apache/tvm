# pylint: disable=invalid-name
"""Namespace of graph pass.

Principle:
- Graph in, graph out: always takes in graph as first argument and returns a graph
- Composable API: break graph transformation pass as segments of small transformations.
"""
from __future__ import absolute_import as _abs
