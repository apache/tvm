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
familiar Python syntax. Decorators acquire the source and its definition context. A syntax
transpiler translates the Python AST into a Python builder program; executing that program
constructs TVM IR. The scripted function body is therefore translated before execution.

TVMScript serves two roles in the TVM stack:

- **Authoring**: users write TIR kernels and Relax programs directly in TVMScript.
- **Roundtrip**: ``mod.script()`` prints IR as TVMScript, which can be re-parsed with any
  required external symbols to reconstruct equivalent IR. This supports inspection,
  debugging, and serialization of printable IR.


Overview
--------

TVMScript separates source acquisition, syntax translation, IR construction, and printing:

.. code-block:: text

   Parsing (Python source → TVM IR):

   Python source (TVMScript)
        │
        ▼  Frontend: source + definition context
        │
   Python AST
        │
        ▼  Syntax transpiler + construction protocol
        │
   Generated Python builder program
        │
        ▼  Frontend: recompose and execute callable
        │
        ▼  IR builders (frame stack + dialect policy)
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

- **Frontend** (Python): captures definition context, acquires source, and composes the
  generated callable with its lexical environment. It owns decorators, helper entry points,
  and execution of the builder program.
- **Syntax transpiler** (Python): rewrites Python AST nodes into calls on the selected
  construction namespace. It preserves source scopes, evaluation order, and locations.
- **IR builders** (Python + C++): own typed values, symbol identity, dialect policy, and
  construction frames. Exiting a frame finalizes its IR and attaches it to its parent.
- **Printer** (C++): converts IR to a ``Doc`` tree and formats it as Python syntax. The
  printer's Doc tree is separate from the Python AST used for parsing.

All source entry points use the canonical ``tvm.script.parser`` implementation.


Decorators
----------

TVMScript uses three import aliases by convention:

.. code-block:: python

   from tvm.script import ir as I       # module-level constructs
   from tvm.script import tirx as T     # TIR constructs
   from tvm.script import relax as R    # Relax constructs
   from tvm.script.parser.entry import make_macro_decorator
   from tvm.target import Target

These are public authoring APIs. ``Target`` configures compilation targets; it remains in
``tvm.target``. Public dialect namespaces expose decorators and source constructors, so
scripted programs do not need direct imports from builder implementation packages.

The primary decorators are:

- ``@I.ir_module``: marks a Python class as an ``IRModule``. Each method inside becomes a
  function in the module.
- ``@T.prim_func``: marks a function as a TIR ``PrimFunc``.
- ``@R.function``: marks a function as a ``relax.Function``.

These can be composed:

.. code-block:: python

   @I.ir_module
   class MyModule:
       @T.prim_func(s_tir=True)
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
               out = R.call_tir(MyModule.add_kernel, (x, y),
                                out_ty=R.Tensor((128,), "float32"))
               R.output(out)
           return out

Python creates the class before invoking ``@I.ir_module``. Function decorators inside an
IR-module class retain method definitions for module construction. The module frontend then
acquires the source and constructs declarations and bodies in the module's builder context.
Standalone ``@T.prim_func`` and ``@R.function`` definitions construct their IR immediately.

A helper can construct expressions or statements in the caller's active builder frames:

.. code-block:: python

   @make_macro_decorator(T)
   def add_one(value):
       return value + 1

Unlike a function decorator, ``make_macro_decorator`` does not create an IR function for
each helper. Its default return behavior is ordinary Python return from the generated
helper program.
``T.inline`` and the dialect macro decorators are supplied through the same frontend helper
mechanism.


Parser Architecture
-------------------

The parser lives in ``python/tvm/script/parser/``. ``inspect_source.py`` owns source
acquisition, the ``Source`` representation and its coordinate helpers, and definition-context
capture. Documented functions in ``entry.py`` manage the parse sequence, callable composition
and execution.
The transpiler handles syntax; the builders interpret concrete values. TIRx JIT validation,
defaults and specialization caching live in ``tvm.tirx.script.jit``; shared parser utilities
remain independent of that policy.

Syntax and construction protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A registered function decorator selects a construction namespace. The same Python syntax
then lowers to operations supplied by that namespace: for example, unmarked ``if`` creates
TIR control-flow frames in a primitive function and Relax control-flow frames in a Relax
function. The transpiler emits these operations without implementing dialect IR semantics.

``parser/protocol_registry.py`` owns callable syntax metadata and registration. Argument policies such
as ``expr_str`` translate symbolic strings written directly in source expressions.
Global-info selectors use the builder decorator
:func:`tvm.script.ir_builder.resolve_global_info_args`, which resolves named string arguments
through an explicit dialect callback after ordinary Python argument evaluation. Captured
aliases and unpacked calls use the same decorator. Dtype and placement strings remain literal. Captured or computed symbolic shapes must already contain explicit IR variables;
the parser does not interpret expression strings found inside captured values.
Registration keeps syntax facts in simple dictionaries keyed by canonical namespace paths.
Known namespace aliases normalize to the registered path; ordinary captured callables,
local aliases, methods and closures execute as Python without marker inference. The
dictionaries retain no source functions or parse captures. Dialects own their namespace
exports, entry points and aliases and register them with the generic parser.
The registry selects a call's argument policy before traversal and reuses positional
parameter names computed once at registration;
``parser/expr_str_handling.py`` decodes annotation/expression strings and maps their
escaped, multiline and UTF-8 source ranges. The main visitor rewrites the decoded syntax.

Assignments become binding operations, standalone expressions become emission operations,
and loops and scopes become builder contexts. An ordinary assignment passes its evaluated
RHS and a separate ``value_span`` to ``bind_``, without first stamping the returned value.
The binding target has its own span. TIRx preserves returned Vars, including BufferVars,
and ordinary metadata objects; other expressions follow the dialect's binding rules.
Construction still runs in the source context, and nested expressions retain their locations.
Concrete binding, type checking, comparison construction, and frame finalization belong to
the builders. Ordinary host calls and
operator overloads execute as part of the generated Python program.

Loop frames create their named variables during construction. Entering a frame returns
the variable directly for one dimension and the variable sequence for multiple dimensions,
so ordinary Python uses ``with T.serial(0, n) as i``. Generated scalar loops use the same
entry form. Generated tuple, list and starred targets unpack the entered frame's stable
``vars`` sequence, preserving explicit one-dimensional unpacking and native variable identity.

Parser syntax restrictions raise ``SyntaxError`` with the original filename and source
range. Python helper and builder exceptions propagate unchanged, preserving their identity,
type and source-mapped traceback. Temporary parser contexts are released on both successful
and exceptional exits.

Parse flow
~~~~~~~~~~

The public entry points include ``tvm.script.parse`` and ``tvm.script.from_source``:

1. **Acquire source and context**: the frontend accepts a function, class, or source string,
   obtains its Python AST, and retains source locations. Decorators capture the definition
   context temporarily through ``definition_scope``; source-string callers can supply
   external bindings through ``extra_vars``. Source inspection belongs to
   ``parser.inspect_source``.
2. **Prescan and translate syntax**: acquisition supplies a fresh owned AST. One
   ``PrescanCollector`` produces read-only ``PrescanContext`` facts for reserved names,
   scoped declarations and conditional outputs. ``IRBuilderTranspiler`` rewrites that owned
   tree with the standard Python AST visitor, using fixed namespace metadata and registered
   argument policies without copying or reparsing the tree. One shared name allocator
   avoids collisions with user identifiers through rewriting and recomposition.
3. **Compose the callable**: the private frontend helper ``_recompose_builder`` compiles
   the generated program with the source's globals, closure bindings, and necessary
   annotation context. Explicit temporary phase inputs connect generated helpers to
   their source functions without rediscovering them through a generated-AST walk.
4. **Execute construction**: the callable enters native builder frames and reconstructs
   declarations and annotations in their required context. When forward declarations are
   needed, it declares all signatures first. Inside each retained native function frame,
   it defines and calls a lexical body helper without an explicit frame argument. Re-entry
   uses the existing parameters instead of adding them again. Native frames own module
   references, results and function-local symbols.
5. **Validate and return IR**: generated code calls the dialect's
   ``X.check_well_formed_`` for a completed function or the shared
   ``I.check_well_formed_`` for a completed module. Public parsing validates by default.
   Parsing owns execution and releases its temporary builder and captures afterward.

Definition and symbol scopes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python lexical bindings and DSL symbolic declarations have different owners. The frontend
captures the exact definition scope during decoration, including enclosing names used only
in annotations that Python may omit from the function's closure. Ordinary body references
retain their source globals and closure bindings. Unrelated callers' local variables do not
become part of this environment.

Annotations are reconstructed as builder operations within the module/function context.
The generated program preserves their definition scope and original names, using a
distinct compiler binding only when a source-local binding would shadow the annotation.
One prescan-seeded name allocator serves rewriting and callable recomposition. Body
globals and closures retain their original lookup, including a method closure whose name
also appears as a class member. Callable recomposition and
lexical environment setup belong to the frontend. The transpiler preserves source scopes
through Python syntax; it does not classify each name by membership in a runtime environment
or search caller stacks for values.

DSL expression strings such as ``R.Tensor(("n", 4), "float32")`` use
``X.resolve_type_var_("n")``, which delegates to the nearest native function frame's
``type_var_map``. Repeated names refer to the same symbol across parameters, return
annotations and the body. Quoted lookup does not introduce a Python name; an explicit
symbol declaration does. A Python name in ``R.Tensor((n, 4), ...)`` follows its source
lexical scope. Relax global-info selectors use ``R.resolve_global_info_`` against the
nearest active native module frame's existing map. String selectors outside an active module
raise, including when an unrelated builder is active. Concrete metadata objects retain identity.
When defining Python functions before their module builder opens, quote the whole annotation
or use ``from __future__ import annotations``; the parser reconstructs these annotations inside
the module. Printed module function signatures containing selectors request postponed annotations.
Eager printed class assignments, such as typed ``R.ExternFunc`` declarations, retrieve concrete
virtual devices with ``R.lookup_vdevice`` because postponed annotations do not defer assignment
values. This explicit lookup preserves the module metadata object before tensor construction.

During rewriting, each needed source range becomes a fixed native span entry in an
injected table. ``_S[i](value)`` attaches that location while preserving identity;
``_S[i].ctx(lambda: helper(value))`` evaluates the callee and arguments once within
the source context and restores it on success or failure. Entries live in
``ir_builder.base`` and retain no AST or active caller context. Dynamic caller/definition
provenance is composed when an entry is used. Ordinary names, Python literals, fixed
namespace/callee lookup and structural indexing tuples need no entries. Arithmetic and
buffer loads retain their own expression ranges.

Generated hooks receive entries explicitly: ``X.for_(X.range_(3), names="i", span=_S[i])``
and ``X.emit_(value, span=_S[j])`` own their result locations. A callable registered with
``result_span`` promises that its complete IR effect is represented by its returned node
or emission receipt, allowing result attachment without a construction context. Opaque
helpers still require ``.ctx`` to locate internal effects. At an ordinary assignment boundary,
``.ctx(..., attach_result=False)`` preserves construction context while leaving the returned
value's attribution to ``bind_`` through its separate ``value_span``.
The parser-owned :mod:`tvm.script.parser.protocol_registry` module defines registration APIs,
metadata and persistent registration state. Dialects apply these policies beside concrete definitions,
constructor creation or necessary exposure sites, without namespace scans or bulk
registration inventories. Dialect ``parser_protocol`` modules implement construction hooks. The shared
:mod:`tvm.script.ir_builder.parser_protocol` documents the detailed builder-hook contract and links
to the registry for special syntax policies. Dialect hooks retain typed signatures and
document only their specific behavior. For example, dialects import ``args_policy``
from ``tvm.script.parser.protocol_registry`` to register literal argument policies.

Ordinary assignments pass returned values to ``bind_``. TIRx Vars keep their exact identity,
producer names and spans, including scope variables, block axes and explicit ``T.bind``
results. This behavior belongs to the builder's value handling and requires no parser
producer classification. Self-emitting builders return ``AlreadyEmitted[T]`` so
source-location handling can annotate the emitted object without emitting it twice.
Layout, meta_class and view results likewise follow dialect value handling for assignment
and standalone statements. Unsupported host values still fail under the dialect's emission
rules. Expression-facing thread APIs return native Vars; IterVar remains native metadata.

Simple comparison chains such as ``0 < i < 10`` lower to
``X.and_(X.lt_(0, i), X.lt_(i, 10))``. Only names and numeric literals, including signed
literals, are supported in a chain. Complex chain operands fail with a source-located
syntax error before they evaluate. Single comparisons retain general operands, and the
explicit constexpr boundary retains Python comparison semantics.

A bare loop target receives the native entry result: ``for iters in T.grid(m, n)`` binds
the original variable sequence, while ``for i in T.serial(n)`` binds one Var. Explicit
tuple, list and starred targets unpack the native frame's stable ``vars`` field.

Function decorators pass their construction namespace and options explicitly to parsing.
Generated ``X.function_(**options)`` creates the frame; the dialect hook owns option defaults.
Declaration creates stable references before function bodies; generated code reads the
native ``global_var`` or ``local_var`` field directly, without a separate reservation pass.
There is no function-registration record or copied source-function metadata. The root
retains ``pyfunc`` behavior and original Python callables for module assembly.

Completed modules use generic root coordination over dialect-owned validation hooks.
Each dialect checks its members and cross-function requirements against the whole module,
including captured members. Shared GlobalInfo construction and lookup belong to ``tvm.ir``;
existing Relax class exports and serialized type keys remain compatible.

Each parse rewrites the fresh AST returned by source acquisition directly, without
copying or reparsing it. Ordinary functions construct arguments directly and bind their
known body parameters from ``frame.params``. Specialization setup and conditional
parameter selection are generated only for an explicit specialization request; an empty
mapping still requests that path, while ``None`` selects ordinary parsing.

Explicit host control flow
~~~~~~~~~~~~~~~~~~~~~~~~~~

Mark compile-time Python selection with ``I.constexpr`` or the identical dialect alias:

.. code-block:: python

   if I.constexpr(enabled):
       T.evaluate(1)
   else:
       T.evaluate(0)

   value = T.ramp(0, 1, lanes) if T.constexpr(lanes > 1) else 0

The marker applies to the controlling value. The generated program evaluates it once and
executes only the selected Python branch in the existing builder scope. Branch bodies and
result expressions still translate normally. Marked ``and`` and ``or`` decisions retain
Python short-circuiting; unmarked conditionals and logical expressions call dialect IR
constructors. A Python boolean alone does not request compile-time branch selection.

JIT supplies validated specialization bindings and optional-argument absence to builder
execution. Syntax translation does not inspect those values to select a branch.


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

   IRBuilder (current construction context)
   └── frame stack:
       ├── IRModuleFrame          ← @I.ir_module
       │   ├── PrimFuncFrame      ← @T.prim_func
       │   │   ├── ForFrame       ← T.grid(...) / T.serial(...)
       │   │   │   └── SBlockFrame ← T.sblock(...)
       │   │   └── ...
       │   └── FunctionFrame      ← @R.function
       │       └── BindingBlockFrame ← R.dataflow()
       └── ...

The generated program constructs IR by entering and exiting frames. Each frame owns its
construction policy and finalization; the syntax transpiler does not maintain an IR tree.

TIR builder
~~~~~~~~~~~

The TIR builder (``python/tvm/tirx/script/ir_builder/``) implements the construction operations
behind public ``T`` syntax. Builder modules also support direct programmatic construction;
public decorators and internal frame-opening operations have distinct roles. Key categories:

**Function and block**:

- ``@T.prim_func`` selects TIR construction; its builder ``function()`` opens ``PrimFuncFrame``
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

The Relax builder (``python/tvm/relax/script/ir_builder/``) implements:

**Function and dataflow**:

- ``@R.function`` selects Relax construction; its builder ``function()`` opens ``FunctionFrame``
- ``R.dataflow()`` → ``BindingBlockFrame``
- ``R.output(*vars)`` → expose variables from a dataflow block

**Emit**:

- ``R.emit(value)`` → emit a binding, returns a ``Var``
- ``R.emit_match_cast(value, ty)`` → emit with type assertion

Generated standalone-statement emission uses the construction protocol's separate ``emit_``
operation. Public ``R.emit`` retains its binding-returning imperative API.

**Type annotations**:

- ``R.Tensor(shape, dtype)`` — tensor type
- ``R.Tuple(*fields)`` — tuple type
- ``R.Shape(values)`` — shape type
- ``R.Any()`` — any Relax value type

**Calling conventions**:

- ``R.call_tir(func, args, out_ty)`` — call a TIR function
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

- ``src/tirx/script/printer/`` — converts PrimFunc, Buffer, SBlock, loops, expressions.
- ``src/relax/script/printer/`` — converts relax.Function, bindings, types, operators.
- ``src/script/printer/ir/`` — converts IRModule, shared types.

The final step calls ``DocToPythonScript()`` (``src/script/printer/doc_printer/python_doc_printer.cc``)
to format the Doc tree into properly indented Python text.

Roundtrip
~~~~~~~~~

For printable IR with the required source context available:

.. code-block:: python

   text = mod.script()           # IR → TVMScript text
   reparsed = tvm.script.from_source(text)  # text → IR
   tvm.ir.assert_structural_equal(mod, reparsed)

This roundtrip property is relied upon by testing infrastructure and serialization workflows.
Printed text uses canonical forms and may differ from hand-written TVMScript. External
objects or Python attachments require their corresponding context; text alone does not
serialize arbitrary Python state.


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
     - No Relax IR loop construction
   * - ``with T.sblock(...)``
     - Spatial block scope
     - N/A
   * - ``with R.dataflow()``
     - N/A
     - Dataflow block
   * - ``if ... else``
     - TIR ``IfThenElse``; explicit ``I.constexpr`` selects a Python branch
     - Relax ``If``; explicit ``I.constexpr`` selects a Python branch
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
     - Type annotation
   * - ``return``
     - Primitive return expression
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

Inside scripted function bodies, unsupported statement forms include ``class``,
``try/except``, ``import``, and ``global``. ``yield`` and ``async/await`` are also unsupported.
Place imports and ordinary Python class definitions in the surrounding Python program.


TIR Syntax Reference
---------------------

Function definition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @T.prim_func
   def func_name(a: T.handle, b: T.handle):
       m = T.int32()
       n = T.int32()
       A = T.match_buffer(a, (m, n), "float32")
       B = T.match_buffer(b, (m,), "float32")
       # function body

- ``T.handle`` — opaque handle parameter (matched to a buffer inside the function).
- ``T.Buffer(shape, dtype)`` — can also be used directly in the signature:
  ``def func(A: T.Buffer((128,), "float32"))``.

Block and axes
~~~~~~~~~~~~~~

Use ``@T.prim_func(s_tir=True)`` for functions with scheduled-TIR blocks and block axes.

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
   T.call_extern("int32", "func_name", *args)                  # external function call
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

- ``R.Tensor(shape, dtype)`` — tensor type annotation.
- ``R.Tuple(...)``, ``R.Shape(...)``, ``R.Any()`` — other Relax type annotations.
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

   out = R.call_tir(MyModule.my_kernel, (x, y), out_ty=R.Tensor((128,), "float32"))

- ``MyModule.my_kernel`` — references a TIR ``PrimFunc`` in the same module.
- ``out_ty`` — the type (shape and dtype) of the output tensor.

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
