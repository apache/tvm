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

.. _tvmscript-arch:

TVMScript
=========

TVMScript is a Python-based domain-specific language (DSL) for writing TVM IR. It lets users
define ``IRModule``\ s — containing both Relax functions and TIR ``PrimFunc``\ s — using
familiar Python syntax. Although TVMScript *looks* like Python, it is **not executed by the
Python interpreter**. Instead, Python decorators extract the AST from the source code and
transform it into TVM IR through a dedicated parser and IR builder pipeline.

TVMScript serves two roles in the TVM stack:

- **Authoring**: users write TIR kernels and Relax programs directly in TVMScript.
- **Roundtrip**: every ``IRModule`` can be printed back to TVMScript via ``mod.script()`` and
  re-parsed to produce an equivalent module. This makes TVMScript the primary tool for
  inspecting, debugging, and serializing IR.


Overview
--------

The TVMScript system has three components:

.. code-block:: text

   Parsing (Python source → TVM IR):

   Python source (TVMScript)
        │
        ▼  ast.parse + convert
        │
   Doc AST (mirror of Python AST)
        │
        ▼  Parser (dispatch by token: ir / tirx / relax)
        │
        ▼  IR Builder (frame stack)
        │
   TVM IR (IRModule, PrimFunc, relax.Function)


   Printing (TVM IR → Python source):

   TVM IR
        │
        ▼  IRDocsifier (C++, dispatch by token + type)
        │
   Doc tree (ExprDoc, StmtDoc, ...)
        │
        ▼  DocToPythonScript
        │
   TVMScript text

- **Parser** (Python): reads Python source, converts it to a ``Doc AST`` (a mirror of
  Python's ``ast`` module), then walks the tree using dialect-specific handlers that call
  into the IR builder.
- **IR Builder** (Python + C++): provides a frame-stack API where each ``with`` block or
  decorator pushes a frame. When the frame exits, the constructed IR is finalized. The builder
  is shared across dialects — TIR and Relax each register their own frame types.
- **Printer** (C++): converts TVM IR objects to a ``Doc`` tree (an intermediate representation
  of Python syntax), then formats the tree into valid TVMScript text.


Decorators
----------

TVMScript uses three import aliases by convention:

.. code-block:: python

   from tvm.script import ir as I       # module-level constructs
   from tvm.script import tirx as T     # TIR constructs
   from tvm.script import relax as R    # Relax constructs

The primary decorators are:

- ``@I.ir_module``: marks a Python class as an ``IRModule``. Each method inside becomes a
  function in the module.
- ``@T.prim_func``: marks a function as a TIR ``PrimFunc``.
- ``@R.function``: marks a function as a ``relax.Function``.

These can be composed:

.. code-block:: python

   @I.ir_module
   class MyModule:
       @T.prim_func
       def add_kernel(A: T.Buffer((128,), "float32"),
                      B: T.Buffer((128,), "float32"),
                      C: T.Buffer((128,), "float32")):
           for i in range(128):
               with T.sblock("compute"):
                   vi = T.axis.spatial(128, i)
                   C[vi] = A[vi] + B[vi]

       @R.function
       def main(x: R.Tensor((128,), "float32"),
                y: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
           with R.dataflow():
               out = R.call_tir(cls.add_kernel, (x, y),
                                out_sinfo=R.Tensor((128,), "float32"))
               R.output(out)
           return out

When Python encounters ``@I.ir_module``, the decorator does **not** execute the class body.
Instead, it calls ``tvm.script.parse()`` which extracts the source code of the class,
builds a Doc AST, and hands it to the parser.


Parser Architecture
-------------------

The parser lives in ``python/tvm/script/parser/``.

Dispatch mechanism
~~~~~~~~~~~~~~~~~~

Different IR dialects (TIR, Relax) need different handling for the same Python syntax. For
example, ``if ... else`` inside ``@T.prim_func`` creates a TIR ``If`` branch, while the same
syntax inside ``@R.function`` creates a Relax ``If`` node with different semantics.

The parser maintains a **dispatch token** stack (``["default"]`` initially). When it encounters
a decorated function, it inspects the decorator to determine the token — ``"tirx"`` for
``@T.prim_func``, ``"relax"`` for ``@R.function`` — and pushes it onto the stack.

Each AST node type is dispatched via a virtual table:

.. code-block:: text

   ParseVTable[(token, node_type)] → handler function

   Lookup order:
     1. (current_token, node_type)    e.g. ("tirx", "For")
     2. ("default", node_type)        e.g. ("default", "For")
     3. generic_visit                  fallback

Dialect-specific parsers (``parser/tirx/parser.py``, ``parser/relax/parser.py``) register
handlers using ``@dispatch.register(token, type_name)`` decorators.

Parse flow
~~~~~~~~~~

The entry point is ``parse(program, extra_vars)``:

1. **Source extraction**: the program's source code is extracted (from a class, function, or
   string) and converted to a Doc AST via Python's ``ast`` module.

2. **AST walking**: the ``Parser`` (a subclass of ``doc.NodeVisitor``) walks the Doc AST.
   For each node, it looks up the handler in the dispatch table.

3. **Expression evaluation**: expressions like ``T.grid(128, 128)`` are evaluated by the
   ``ExprEvaluator``, which resolves names against the variable table and the ``T.``/``R.``
   module namespaces.

4. **Value binding**: assignment statements (``A = T.match_buffer(...)`` in TIR,
   ``lv = R.add(x, y)`` in Relax) go through dialect-specific ``bind_*_value()`` functions
   that register the resulting TVM objects in the parser's ``VarTable``.

5. **Scoping**: the ``VarTable`` maintains a stack of frames. Entering a ``with`` block,
   ``for`` loop, or function body pushes a new frame; exiting pops it. This ensures variables
   are scoped correctly.

Variable table
~~~~~~~~~~~~~~

The ``VarTable`` is the parser's symbol table:

.. code-block:: text

   VarTable
   ├── frames: [VarTableFrame, ...]    ← stack of scopes
   └── name2value: {str: [Any, ...]}   ← name → value stack (for shadowing)

When a name is looked up, the most recent binding wins. When a frame is popped, all bindings
introduced in that frame are removed.


IR Builder Architecture
-----------------------

The IR builder (``python/tvm/script/ir_builder/``, backed by C++ in ``src/script/ir_builder/``)
provides a frame-stack API for constructing IR incrementally.

Frame stack
~~~~~~~~~~~

The core idea: each IR scope (module, function, block, loop) is a **frame**. Frames are pushed
on ``__enter__`` and popped on ``__exit__``. When a frame exits, it finalizes the IR it
represents and attaches it to the parent frame.

.. code-block:: text

   IRBuilder (thread-local singleton)
   └── frame stack:
       ├── IRModuleFrame          ← @I.ir_module
       │   ├── PrimFuncFrame      ← @T.prim_func
       │   │   ├── ForFrame       ← T.grid(...) / T.serial(...)
       │   │   │   └── SBlockFrame ← T.sblock(...)
       │   │   └── ...
       │   └── FunctionFrame      ← @R.function
       │       └── BindingBlockFrame ← R.dataflow()
       └── ...

This design means the parser never needs to build a complete IR tree in memory — it
constructs IR top-down by entering and exiting frames, and each frame handles its own
finalization.

TIR builder
~~~~~~~~~~~

The TIR builder (``ir_builder/tirx/ir.py``) provides functions that map directly to TVMScript
syntax. Key categories:

**Function and block**:

- ``T.prim_func()`` → ``PrimFuncFrame``
- ``T.sblock(name)`` → ``SBlockFrame`` (spatial block)
- ``T.init()`` → ``BlockInitFrame`` (reduction initialization)
- ``T.reads(...)``, ``T.writes(...)`` → declare buffer access regions

**Loops**:

- ``T.grid(*extents)`` → ``ForFrame`` returning loop variables
- ``T.serial(start, stop)``, ``T.parallel(...)``, ``T.vectorized(...)``,
  ``T.unroll(...)``, ``T.thread_binding(...)`` → loop with specific iterator type

**Block axes**:

- ``T.axis.spatial(dom, binding)`` — spatial iteration axis
- ``T.axis.reduce(dom, binding)`` — reduction axis
- ``T.axis.remap(kinds, bindings)`` — shorthand for multiple axes

**Buffers**:

- ``T.match_buffer(param, shape, dtype)`` — match function parameter to buffer
- ``T.alloc_buffer(shape, dtype)`` — allocate intermediate buffer
- ``T.Buffer(shape, dtype)`` — buffer type annotation in function signatures

Relax builder
~~~~~~~~~~~~~

The Relax builder (``ir_builder/relax/ir.py``) provides:

**Function and dataflow**:

- ``R.function()`` → ``FunctionFrame``
- ``R.dataflow()`` → ``BindingBlockFrame``
- ``R.output(*vars)`` → expose variables from a dataflow block

**Emit**:

- ``R.emit(value)`` → emit a binding, returns a ``Var``
- ``R.emit_match_cast(value, struct_info)`` → emit with type assertion

**Type annotations**:

- ``R.Tensor(shape, dtype)`` — tensor struct info
- ``R.Tuple(*fields)`` — tuple struct info
- ``R.Shape(values)`` — shape struct info
- ``R.Object()`` — opaque object struct info

**Calling conventions**:

- ``R.call_tir(func, args, out_sinfo)`` — call a TIR function
- ``R.call_packed(name, *args)`` — call a PackedFunc
- ``R.call_dps_packed(func, *args)`` — call using destination-passing style

**Operators**: the ``R`` module also re-exports all Relax operators
(``R.add``, ``R.matmul``, ``R.nn.conv2d``, etc.) so they can be used directly in TVMScript.


Printer Architecture
--------------------

The printer converts TVM IR back to TVMScript text. It is implemented primarily in C++
(``src/script/printer/``) for performance.

Doc tree
~~~~~~~~

The printer does **not** generate text directly. Instead, it first builds a ``Doc`` tree — an
intermediate representation that mirrors Python syntax:

- **Expression docs**: ``IdDoc``, ``AttrAccessDoc``, ``CallDoc``, ``IndexDoc``,
  ``OperationDoc``, ``LiteralDoc``, ``TupleDoc``, ``ListDoc``, etc.
- **Statement docs**: ``AssignDoc``, ``ForDoc``, ``IfDoc``, ``ScopeDoc`` (``with`` blocks),
  ``FunctionDoc``, ``ClassDoc``, ``ReturnDoc``, ``CommentDoc``, etc.

For example, ``T.axis.spatial(128, i)`` is represented as:

.. code-block:: text

   CallDoc(
     callee=AttrAccessDoc(AttrAccessDoc(IdDoc("T"), "axis"), "spatial"),
     args=[LiteralDoc(128), IdDoc("i")]
   )

IRDocsifier
~~~~~~~~~~~

The ``IRDocsifier`` (``include/tvm/script/printer/ir_docsifier.h``) is the main dispatcher.
It maintains:

- A dispatch table mapping ``(token, type_index)`` pairs to converter functions.
- A frame stack for tracking the current scope (similar to the builder's frame stack).
- A variable-to-name mapping to produce readable names.

Each IR dialect registers its own converters:

- ``src/script/printer/tirx/`` — converts PrimFunc, Buffer, SBlock, loops, expressions.
- ``src/script/printer/relax/`` — converts relax.Function, bindings, struct info, operators.
- ``src/script/printer/ir/`` — converts IRModule, shared types.

The final step calls ``DocToPythonScript()`` (``src/script/printer/doc_printer/python_doc_printer.cc``)
to format the Doc tree into properly indented Python text.

Roundtrip guarantee
~~~~~~~~~~~~~~~~~~~

For any ``IRModule`` constructed through the compiler:

.. code-block:: python

   text = mod.script()           # IR → TVMScript text
   reparsed = tvm.script.from_source(text)  # text → IR
   tvm.ir.assert_structural_equal(mod, reparsed)

This roundtrip property is relied upon by testing infrastructure and serialization workflows.
Note that the printed text may differ from hand-written TVMScript — the printer uses canonical
forms (e.g., explicit ``R.emit`` calls, fully qualified buffer annotations) that are not required
in hand-written code.


Supported Python Syntax
-----------------------

TVMScript supports a subset of Python syntax. The table below summarizes what is supported
and how each construct is interpreted:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Python Syntax
     - TIR
     - Relax
   * - ``for i in range(n)``
     - Serial loop nest
     - Not supported (no Relax-level ``for`` handler)
   * - ``with T.sblock(...)``
     - Spatial block scope
     - N/A
   * - ``with R.dataflow()``
     - N/A
     - Dataflow block
   * - ``if ... else``
     - TIR ``If`` branch (PrimExpr condition) or static eval (Python bool)
     - Relax ``If`` node (plain Python ``if cond:`` syntax)
   * - ``while``
     - ``T.While`` loop
     - Not supported
   * - ``x = expr``
     - Variable binding
     - Emit binding (implicit ``R.emit``)
   * - ``x: T.Buffer(...)``
     - Buffer annotation
     - N/A
   * - ``x: R.Tensor(...)``
     - N/A
     - Struct info annotation
   * - ``return``
     - Not used
     - Function return value
   * - ``A[i, j]``
     - Buffer load
     - Not applicable (use operators)
   * - ``A[i, j] = expr``
     - Buffer store
     - Not applicable
   * - Arithmetic (``+``, ``-``, etc.)
     - PrimExpr operations
     - Calls to Relax operators
   * - Function calls
     - ``T.*`` intrinsics
     - ``R.*`` operators or ``call_tir`` / ``call_packed``

**Not supported**: ``class`` definitions (except for ``@I.ir_module``), ``try/except``,
``yield``, ``async/await``, list comprehensions, ``lambda``, ``import``, and ``global``
statements.


TIR Syntax Reference
---------------------

Function definition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @T.prim_func
   def func_name(a: T.handle, b: T.handle):
       A = T.match_buffer(a, (m, n), "float32")
       B = T.match_buffer(b, (m,), "float32")
       # function body

- ``T.handle`` — opaque handle parameter (matched to a buffer inside the function).
- ``T.Buffer(shape, dtype)`` — can also be used directly in the signature:
  ``def func(A: T.Buffer((128,), "float32"))``.

Block and axes
~~~~~~~~~~~~~~

.. code-block:: python

   for i, j in T.grid(128, 128):
       with T.sblock("block_name"):
           vi = T.axis.spatial(128, i)
           vj = T.axis.reduce(128, j)
           T.reads(A[vi, vj])
           T.writes(B[vi])
           # compute

- ``T.axis.spatial`` / ``T.axis.reduce`` / ``T.axis.scan`` — declare axis variables with
  their iteration domain and binding to outer loop variables.
- ``T.axis.remap("SR", [i, j])`` — shorthand: ``S`` = spatial, ``R`` = reduce.
- ``T.reads(...)``, ``T.writes(...)`` — declare buffer regions accessed by this block.

Loop types
~~~~~~~~~~

.. code-block:: python

   for i in T.serial(0, 128):          # sequential
   for i in T.parallel(0, 128):        # parallel
   for i in T.vectorized(0, 128):      # vectorized
   for i in T.unroll(0, 128):          # unrolled
   for i in T.thread_binding(0, 128, thread="threadIdx.x"):  # GPU thread

Buffer operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   C = T.alloc_buffer((128, 128), "float32")  # intermediate buffer
   val = A[i, j]                               # buffer load
   B[i] = val + 1.0                            # buffer store

Common intrinsics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   T.exp(x), T.log(x), T.sqrt(x), T.tanh(x), ...  # math functions
   T.cast(x, "float16")                              # type cast
   T.if_then_else(cond, true_val, false_val)          # conditional expression
   T.min(a, b), T.max(a, b)                           # min/max
   T.call_extern("func_name", *args)                  # external function call
   T.call_packed("func_name", *args)                   # packed function call
   T.tvm_storage_sync("shared")                        # GPU memory fence


Relax Syntax Reference
-----------------------

Function definition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @R.function
   def main(x: R.Tensor((128, 128), "float32"),
            y: R.Tensor((128,), "float32")) -> R.Tensor((128, 128), "float32"):
       # function body
       return result

- ``R.Tensor(shape, dtype)`` — tensor type annotation (struct info).
- ``R.Tuple(...)``, ``R.Shape(...)``, ``R.Object()`` — other struct info types.
- ``R.function(private=True)`` — marks the function as module-private.
- ``R.function(pure=False)`` — marks the function as having side effects.

Dataflow blocks
~~~~~~~~~~~~~~~

.. code-block:: python

   with R.dataflow():
       lv0 = R.add(x, y)
       lv1 = R.nn.relu(lv0)
       R.output(lv1)

Variables inside a ``R.dataflow()`` block are local to that block. ``R.output(...)`` exposes
variables to the outer scope.

Calling TIR functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   out = R.call_tir(cls.my_kernel, (x, y), out_sinfo=R.Tensor((128,), "float32"))

- ``cls.my_kernel`` — references a TIR ``PrimFunc`` in the same module.
- ``out_sinfo`` — the struct info (shape and dtype) of the output tensor.

Control flow
~~~~~~~~~~~~

Relax ``if`` uses plain Python ``if`` syntax. The condition must be a Relax variable with
boolean type. Both branches are required.

.. code-block:: python

   @R.function
   def f(cond: R.Tensor((), "bool"), x: R.Tensor((128,), "float32")):
       if cond:
           result = R.add(x, x)
       else:
           result = R.multiply(x, x)
       return result


Source Code Map
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Path
     - Contents
   * - ``python/tvm/script/parser/core/``
     - Core parser: dispatch, expression evaluator, variable table, Doc AST
   * - ``python/tvm/script/parser/tirx/``
     - TIR-specific parser handlers and value binding
   * - ``python/tvm/script/parser/relax/``
     - Relax-specific parser handlers and value binding
   * - ``python/tvm/script/parser/ir/``
     - ``@I.ir_module`` entry point and module-level parsing
   * - ``python/tvm/script/ir_builder/base.py``
     - IRBuilder base class and frame stack mechanism
   * - ``python/tvm/script/ir_builder/tirx/``
     - TIR frame types and builder functions (``T.*``)
   * - ``python/tvm/script/ir_builder/relax/``
     - Relax frame types and builder functions (``R.*``)
   * - ``python/tvm/script/ir_builder/ir/``
     - IRModule builder (``I.*``)
   * - ``src/script/printer/``
     - C++ printer: Doc tree, IRDocsifier, Python code generation
   * - ``src/script/printer/tirx/``
     - TIR-specific IR-to-Doc converters
   * - ``src/script/printer/relax/``
     - Relax-specific IR-to-Doc converters
   * - ``src/script/ir_builder/``
     - C++ backend for frame stack and IR construction
   * - ``include/tvm/script/printer/``
     - C++ headers: Doc classes, IRDocsifier, dispatch functor
   * - ``include/tvm/script/ir_builder/``
     - C++ headers: builder base, dialect-specific frame types
