..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _hybrid-langref-label:

Hybrid Frontend Language Reference
==================================

Overview
--------

This hybrid frontend allows users to write preliminary versions of some idioms that yet have
been supported by TVM officially.

Features
--------

Software Emulation
~~~~~~~~~~~~~~~~~~

Both software emulation and compilation are supported. To define a function,
you need to use ``tvm.te.hybrid.script`` decorator to indicate this is a hybrid function:

.. code-block:: python

    @tvm.te.hybrid.script
    def outer_product(a, b):
        c = output_tensor((100, 99), 'float32')
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                c[i, j] = a[i] * b[j]
        return c
    a = numpy.random.randn(100)
    b = numpy.random.randn(99)
    c = outer_product(a, b)


This decorator will import `Keywords`_ required spontaneously when software emulation.
After software emulation is done, the imported keywords will be cleaned up. Users do not need
worry about keyword conflict and pollution.

Every element passed for software emulation in the argument list is either a python variable
or ``numpy`` numeric type.

Backend Compilation
~~~~~~~~~~~~~~~~~~~

This function is not encouraged to use, users are encouraged to use the second interface.
The current parse interface looks like:

.. code-block:: python

   a = tvm.te.placeholder((100, ), name='a')
   b = tvm.te.placeholder((99, ), name='b')
   parser = tvm.hybrid.parse(outer_product, [a, b]) # return the parser of this function


If we pass these tvm data structures, like ``Tensor``, ``Var``, ``Expr.*Imm``,
or ``tvm.container.Array``, to this function, it returns a op node:

.. code-block:: python

   a = tvm.te.placeholder((100, ), name='a')
   b = tvm.te.placeholder((99, ), name='b')
   c = outer_product(a, b) # return the output tensor(s) of the operator

You can use any methods that can be applied on a TVM ``OpNode``, like create_schedule, although
so far, the functionality of schedule is as limited as ``ExternOpNode``. At least, it can be built
to LLVM module.

Tuning
~~~~~~

Follow up the example above, you can use some tvm like interfaces to tune the code:

.. code-block:: python

   i, j = c.op.axis
   sch = te.create_schedule(op)
   jo, ji = sch.split(j, 4)
   sch.vectorize(ji)

For now, you can use loop annotations (``unroll``, ``parallel``, ``vectorize``, and ``bind``),
loop manipulation (``split`` and ``fuse``), and ``reorder``.

.. note::

        This is a preliminary function, so users should be in charge of the correctness
        of the functionality after tuning. Specifically, users should be careful when
        fusing and reorderding imperfect loops.

Loops
~~~~~

In HalideIR, loops have in total 4 types: ``serial``, ``unrolled``, ``parallel``, and ``vectorized``.

Here we use ``range`` aka ``serial``, ``unroll``, ``parallel``, and ``vectorize``,
these **4** keywords to annotate the corresponding types of for loops.
The usage is roughly the same as Python standard ``range``.

Besides all the loop types supported in Halide, ``const_range`` is supported for some specific conditions.
Sometimes, ``tvm.container.Array`` is desired to pass as an argument, but in TVM-HalideIR, there is no
such support that converts ``tvm.container.Array`` to an ``Expr``. Thus, a limited feature is supported.
Users can access containers by either constants or constants loops annotated.

.. code-block:: python

   @tvm.te.hybrid.script
   def foo(a, b): # b is a tvm.container.Array
       c = output_tensor(a.shape, a.dtype)
       for i in const_range(len(a)): # because you have b access, i should be explicitly annotated as const_range
           c[i] = a[i] + b[i]
       return c


Variables
~~~~~~~~~

All the mutable variables will be lowered to an array with size 1.
It regards the first store of a variable as its declaration.

.. note::

        Unlike conventional Python, in hybrid script, the declared variable
        can only be used in the scope level it is declared.


.. note::

        Currently, you can ONLY use basic-typed variables, i.e. the type of the
        variable should be either ``float32``, or ``int32``.

.. code-block:: python

   for i in range(5):
       s = 0 # declaration, this s will be a 1-array in lowered IR
       for j in range(5):
         s += a[i, j] # do something with s
       b[i] = s # you can still use s in this level
   a[0] = s # you CANNOT use s here, even though it is allowed in conventional Python


Attributes
~~~~~~~~~~

So far, ONLY tensors' ``shape`` and ``dtype`` attribute are supported!
The ``shape`` attribute is essentially a tuple, so you MUST access it as an array.
Currently, only constant-indexed access is supported.

.. code-block:: python

   x = a.shape[2] # OK!
   for i in range(3):
      for j in a.shape[i]: # BAD! i is not a constant!
          # do something


Conditional Statement and Expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   if condition1 and condition2 and condition3:
       # do something
   else:
       # do something else
   # Select
   a = b if condition else c

However, NO ``True`` and ``False`` keyword supported yet.


Math Intrinsics
~~~~~~~~~~~~~~~

So far, these math intrinsics, ``log``, ``exp``, ``sigmoid``,
``tanh``, ``power``, and ``popcount``, are supported.
No import is required, just as it is mentioned in `Software Emulation`_, just use it!

Array Allocation
~~~~~~~~~~~~~~~~

**Under construction, this function will be supported later!**

Use a function call ``allocation(shape, type, share/local)`` to declare an array buffer.
The basic usage is roughly the same as a normal ``numpy.array``, and you should access
high-dim array in ``a[i, j, k]`` fashion instead of ``a[i][j][k]``,
even for ``tvm.container.Array`` for compilation.


Thread Bind
~~~~~~~~~~~


You can also do loop-thread bind by writing code like this:

.. code-block:: python

   for tx in bind("threadIdx.x", 100):
       a[tx] = b[tx]


Assert Statement
~~~~~~~~~~~~~~~~

Assert statement is supported, you can simply use it as it is in standard Python.

.. code-block:: python

    assert cond, mesg

.. note::

        ``Assert`` is NOT a function call. Users are encouraged to use assert in the way
        presented above --- condition followed by message. It fits both Python AST and HalideIR.

Keywords
~~~~~~~~
- For keywords: ``serial``, ``range``, ``unroll``, ``parallel``, ``vectorize``, ``bind``, ``const_range``
- Math keywords: ``log``, ``exp``, ``sqrt``, ``rsqrt``, ``sigmoid``, ``tanh``, ``power``, ``popcount``, ``round``, ``ceil_div``
- Allocate keywords: ``allocate``, ``output_tensor``
- Data type keywords: ``uint8``, ``uint16``, ``uint32``, ``uint64``, ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``float32``, ``float64``
- Others: ``max_num_threads``
