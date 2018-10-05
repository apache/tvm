"""
Writing a Customized Pass
=========================
**Author**: `Jian Weng <https://were.github.io>`_

TVM is a framework that abstracts away the heterogenity of machine learning accelerators.
Sometimes users may want customize some analysis and IR transformations
to adapt TVM to their own specialized hardware. This tutorial helps users write
a customized pass in TVM.

Prerequisites
-------------

Before reading this tutorial, we assume readers have already known these topics well:

- Writing an algorithm in TVM and schedule it. Otherwise, see example tutorials like
  :ref:`opt-gemm`.
- The basic structure of HalideIR. Otherwise, see ``HalideIR/src/ir/IR.h`` to learn what
  attributes of IR nodes are defined.
- Visitor design pattern. Otherwise, check the
  `Python AST module <https://docs.python.org/3/library/ast.html>`_ to see how an AST
  visitor is implemented.
- How a HalideIR/Schedule is lowered to either a LoweredFunc class or a LLVM module. Otherwise,
  take a look at ``python/tvm/build_module.py`` to get some basics.

"""

from __future__ import absolute_import, print_function
import tvm
import numpy as np

######################################################################
# We first write a very simple vector add and build it with the default schedule. Then, we use
# our customized lowering pass to manipulate the IR directly instead of using schedule premitives.
#

n = tvm.const(128)
a = tvm.placeholder((n, ), name="a")
b = tvm.placeholder((n, ), name="b")
c = tvm.compute((n, ), lambda i: a[i] + b[i], name='c')

sch = tvm.create_schedule(c.op)
ir  = tvm.lower(sch, [a, b, c], simple_mode=True)
print(ir)

######################################################################
# Writing a Pass
# --------------
# Essentially, an "IR transformation pass" is a function which maps a statement to a new statement.
# Thus, we define this vectorize function and implement it step by step.
#

######################################################################
# TVM already provides two class for users to both analyze and transform IR.
#
# IR Visitor
# ~~~~~~~~~~
# We can use ``tvm.ir_pass.PostOrderVisit(stmt, func)`` to gather information from the Halide IR.
# ``func`` is a function callback. This function will be called before exiting the current IR node,
# i.e. post-order visit. Then we leverage side effects to store the result of IR visit, because the
# return value of ``func`` will be ignored.
#
# .. note::
#
#     You MUST use some array to store the result of IR visit. Even the value is a single variable.
#     This is mainly due to the constraints in the Python-C runtime. The variable values will be
#     refreshed every recursion but the array values will be preserved.
#

loops = []
def find_width8(op):
    """ Find all the 'For' nodes whose extent can be divided by 8. """
    if isinstance(op, tvm.stmt.For):
        if isinstance(op.extent, tvm.expr.IntImm):
            if op.extent.value % 8 == 0:
                loops.append(op)

#####################################################################
# IR Transformation
# ~~~~~~~~~~~~~~~~~
# The transformation interface is slightly different from the visitor interface. There is only a
# post-order callback in the visitor, but transformation visitor supports both a pre-order and a
# post-order callback. If you want to keep the origin IR node, just return None. If you want to
# change the current node to some node, use TVM IR maker interface to build it and return
# this value.
#
# .. note::
#
#     If the pre-order function is called and returns a value which is not None, the post-order
#     function will be skipped.
#

def vectorize8(op):
    """ Split can vectorize the loops found in `find_width8`. """
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = tvm.var(name + '.outer'), tvm.var(name + '.inner')
        body = tvm.ir_pass.Substitute(op.body, {op.loop_var: lo * 8 + li})
        body = tvm.make.For(li, 0, 8, tvm.stmt.For.Vectorized, 0, body)
        body = tvm.make.For(lo, 0, extent // 8, tvm.stmt.For.Serial, 0, body)
        return body
    return None

def vectorize(stmt):
    global loops

    tvm.ir_pass.PostOrderVisit(stmt, find_width8)

    if not loops:
        return stmt

    # The last list arugment indicates what kinds of nodes will be transformed.
    # Thus, in this case only `For` nodes will call `vectorize8`
    stmt = tvm.ir_pass.IRTransform(stmt, None, vectorize8, ['For'])

    return stmt

#####################################################################
# Glue to Lowering
# ----------------
# So far, we are done with writing this IR transformation pass. What we need to do next is to glue
# this pass to TVM's lower pass. We can first call this function directly as a sanity check.
#

print(vectorize(ir))

#####################################################################
# In TVM, there is a property called ``BuildConfig``. You can use this property to customize your
# own lowering options. In this case, we inject the pass written above into the TVM standard lowering
# pass by feeding **a list of tuple** as argument to ``add_lower_pass``. "Tuple" indicates different
# phases of lowering. In TVM, there are four phases of lowering and user-customized ones will be
# called after each phase is done.
#
# .. note::
#     Here are the essential transformations done by each phase:
#       - Phase 0 generates the raw IR and loop levels.
#       - Phase 1 flattens the array storage.
#       - Phase 2 transforms loops, like unroll, vectorization and thread-binding.
#       - Phase 3 does some cleanup work.
#
# Thus, a good place to put this transformation pass is just after Phase 1.
#

with tvm.build_config(add_lower_pass=[(1, vectorize)]) as cfg:
    print(tvm.lower(sch, [a, b, c], simple_mode=True))

#####################################################################
# Quick View
# ----------
# This tutorial gives a quick view of writing a customized IR transformation pass:
# - Use ``tvm.ir_pass.PostOrderVisit`` to gather information on each IR nodes.
# - Use ``tvm.ir_pass.IRTransform`` to transform IR nodes.
# - Wrap up two above to write an IR-transformation function.
# - Use ``tvm.build_config`` to put this function to TVM lowering pass
#
