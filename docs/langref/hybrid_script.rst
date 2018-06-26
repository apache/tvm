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
you need to use ``tvm.hybrid.script`` decorator to indicate this is a hybrid function:

.. code-block:: python

    @tvm.hybrid.script
    def outer_product(a, b, c):
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                c[i, j] = a[i] * b[j]
    a = numpy.random.rand(100)
    b = numpy.random.rand(99)
    c = numpy.zeros((100, 99))
    outer_product(a, b, c)

This decorator will import `Keywords`_ required spontaneously when software emulation.
After software emulation is done, the imported keywords will be cleaned up. Users do not need
worry about keyword conflict and pollution.

Every element passed for software emulation in the argument list is either a python variable
or ``numpy`` numeric type.

Backend Compilation
~~~~~~~~~~~~~~~~~~~

The current parse interface looks like:

.. code-block:: python

   a = tvm.placeholder((100, ), name='a')
   b = tvm.placeholder((99, ), name='b')
   c = tvm.placeholder((100, 99), name='c')
   tvm.hybrid.parse(outer_product, [a, b, c]) # return an ir root of this function

If we pass these tvm tensors to this function, it returns a op node:

**Under construction, we are still deciding what kind of node should be returned.**

.. code-block:: python

   a = tvm.placeholder((100, ), name='a')
   b = tvm.placeholder((99, ), name='b')
   c = tvm.placeholder((100, 99), name='c')
   op = outer_product(a, b, c) # return the corresponding op node

Tuning
~~~~~~

**Under construction, not truly supported yet.**

Follow up the example above, you can use some tvm like interfaces to tune the code: 

.. code-block:: python

   sch = tvm.create_schedule(op)
   jo, ji = sch.split(j, 4)
   sch.vectorize(ji)

``split``, ``reorder``, and loop_annotation will be supported!

Loops
~~~~~

In HalideIR, loops have in total 4 types: ``serial``, ``unrolled``, ``parallel``, and ``vectorized``.

Here we use ``range`` aka ``serial``, ``unroll``, ``parallel``, and ``vectorize``,
these **4** keywords to annotate the corresponding types of for loops.
The the usage is roughly the same as Python standard ``range``.

Variables
~~~~~~~~~

All the mutatable variables will be lowered to an array with size 1.
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
     	  s += a[i, j] # do something with sum
       b[i] = sum # you can still use sum in this level
   a[0] = s # you CANNOT use s here, even though it is allowed in conventional Python
   b = (1, 2) # this has NOT been supported yet!


Attributes
~~~~~~~~~~

So far, ONLY tensors' ``shape`` attribute is supported! The ``shape`` atrribute is essentailly a
tuple, so you MUST access it as an array. Also, currently, only constant-indexed access is supported.

.. code-block:: python

   x = a.shape[2] # OK!
   for i in range(3):
      for j in a.shape[i]: # BAD! i is not a constant!
          # do something


Conditional Statement and Expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   if condition:
        # do something
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
The basic usage is roughly the same as a normal array.


Thread Bind
~~~~~~~~~~~


You can also do loop-thread bind by writing code like this:

.. code-block:: python

   for tx in bind("threadIdx.x", 100):
       a[tx] = b[tx]


Keywords
~~~~~~~~
- For keywords: ``serial``, ``range``, ``unroll``, ``parallel``, ``vectorize``, ``bind``
- Math keywords: ``log``, ``exp``, ``sigmoid``, ``tanh``, ``power``, ``popcount``
