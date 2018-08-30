"""
Writing a Customized Pass
=========================
**Author**: `Jian Weng <https://were.github.io>`_

TVM is a framework to abstract the heterogenity of those various machine learning
accelerators. Sometimes users may want to customized some analysis and IR transformation
to adopt TVM to their own specialized hardware. This tutorial helps users write
a customized pass in TVM.
 Prerequisites
-------------
Before reading this tutorial, we assume readers have already known these well:
- Writing an algorithm in TVM and schedule it. If not, you should go through other
  tutorials first.
- The basic structure of HalideIR. If not, you should go to ``HalideIR/src/ir/IR.h``
  to see what attributes of IR nodes are defined.
- Visitor design pattern. If not, you can go to Python ``ast`` module to see how an AST
  visitor is implemented.
- How a HalideIR/Schedule is lowered to either a LoweredFunc class or a LLVM module. If
  not, you can go to ``python/tvm/build_module.py`` to get some basic idea about it.
If all these above are true for you. Import these header and let us start!
"""

from __future__ import absolute_import, print_function
import tvm
import numpy as np

######################################################################
# We first write a very simple vector add and build it with the default schedule. Then, we use
# TVM interfaces to manipulate the IR manually instead of using those schedule premitives.  
#

n = tvm.const(128)
a = tvm.placeholder((n, ), name="a")
b = tvm.placeholder((n, ), name="b")
c = tvm.compute((n, ), lambda i: a[i] + b[i], name='c')

sch = tvm.create_schedule(c.op)
ir  = tvm.lower(sch, [a, b, c], simple_mode=True)
print(ir)

