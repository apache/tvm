Hybrid Frontend Developer Guide
===============================

If you are a developer:

1. who is trying writing some preliminary patterns that have not been supported by TVM yet,
maybe :ref:`hybrid-langref-label` is a better place for you.

2. who wants to know the implementing details of this module, you are right here!

Features
--------

Software emulation
~~~~~~~~~~~~~~~~~~

In software emulation, the most intresting thing is the decorator ``tvm.hybrid.script``.
This decorator helps 2 things:

1. Importing runtime variables

2. Overload the function according to the arguments passed

Correct me if I am wrong: I believe that how 1. is implemented is dangerous, but I have no
choice. What I did is add those names into python dict ``func.__global__`` and after
the call to ``func`` is done, those names will be cleaned up. 

Overload is simple: the decorator checks the arguments' types and determines which function
should be actually called.


Backend Compilation
~~~~~~~~~~~~~~~~~~~

Compilation is a large module, you can see ``python/tvm/hybrid/var_decl.py`` and
``python/tvm/hybrid/parser.py`` for more details. The first stage determines the
usage, or more accurately the declaration of each variable and the second stage does
the actual IR generation.

Attributes
~~~~~~~~~~

So far, ONLY tensors' `shape` attribute is supported. You can see ``visit_Subscript``
in ``python/tvm/hybrid/parser.py`` for more details. This is a hacky solution, I just
check the attributes when subscript.

Loops
~~~~~

In HalideIR, loops have in total 4 types: ``serial``, ``unrolled``, ``parallel``, and ``vectorized``.


.. note::

    Unlike what that is in HalideIR, in ``loop_type(a, b)``, ``a`` is the starting point and ``b``
    is the trip count of iterations. Here ``loop_type(a, b)`` indicates ``[a, b)``. Thus, when lowering it
    to HalideIR, we need to do ``start, extent = a, b - a``


.. note::

    In HalideIR those are enums, they are in passive form.
    Here we use active form to annotate loops, because they are ready to run.


Variables
~~~~~~~~~

Because there is no variables in ``HalideIR``, all the mutatable variables will be lowered to an array with size 1.
It takes the first store of a variable as its declaration.

Math intrinsics
~~~~~~~~~~~~~~~
So far, these math intrinsics, ``log``, ``exp``, ``sigmoid``, ``tanh``, ``power``, and ``popcount``, are supported.
Math intrinsics will be imported by the decorator. Most of the intrinsics are borrowed by library implementation
except ``popcount`` and ``sigmoid``. I implemented them manually.
