"""Store for onnx examples and common models."""
from __future__ import absolute_import as _abs
import os
from .super_resolution import get_super_resolution

__all__ = ['super_resolution']

def _as_abs_path(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir, fname)

# a pair of onnx pb file and corresponding nnvm symbol
super_resolution = (_as_abs_path('super_resolution.onnx'), get_super_resolution())
