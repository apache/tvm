"""
Get Started with TVM Go
=======================
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
tgt="llvm"

######################################################################
# Describe the Computation
# ------------------------
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

######################################################################
# Schedule the Computation
# ------------------------
s = tvm.create_schedule(C.op)

######################################################################
# Compilation
# -----------
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

######################################################################
# Save Compiled Module
# --------------------
from tvm.contrib import cc
from tvm.contrib import util

fadd.save("deploy.o")
cc.create_shared("deploy.so", ["deploy.o"])
