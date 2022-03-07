# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Get Started with TVM Go
=======================
"""
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

# Global declarations of environment.

tgt = "llvm"

######################################################################
# Describe the Computation
# ------------------------
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

######################################################################
# Schedule the Computation
# ------------------------
s = te.create_schedule(C.op)

######################################################################
# Compilation
# -----------
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

######################################################################
# Save Compiled Module
# --------------------
from tvm.contrib import cc
from tvm.contrib import utils

fadd.save("deploy.o")
cc.create_shared("deploy.so", ["deploy.o"])
