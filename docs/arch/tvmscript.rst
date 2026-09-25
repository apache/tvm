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

TVMScript expresses TVM IR with Python syntax. A shared frontend translates source
into a Python builder program; executing that program constructs IR. A separate
printer converts IR back into readable, parseable source.

.. code-block:: text

   Source + definition context
       → Python AST → syntax transpiler → Python builder program → TVM IR
   TVM IR
       → IRDocsifier → Doc tree → Python text

Source and frontend
-------------------

``tvm.script.parser`` accepts Python functions, classes and source strings through
public decorators, ``tvm.script.parse`` and ``tvm.script.from_source``. The frontend
acquires source and locations, captures the definition's globals and closure bindings,
and composes the generated callable with that environment. String callers provide
external bindings through ``extra_vars``.

Standalone function decorators construct IR immediately. Within an IR-module class,
function decorators retain definitions until module construction, which can declare
signatures before building bodies. Captured Python values belong to the source
context; symbolic IR values are created and resolved by builders.

Annotations read concrete values from their definition scope, preserving missing-name
errors when a value is used. Create external symbols with ``n = I.dynamic("n")``
or the identical ``T.dynamic`` and ``Ts.dynamic`` constructors. Each call creates a
fresh native variable, defaulting to int64. On Python 3.12+, explicit headers such as
``def f[n, k: T.int32](...)`` declare local symbols; ``n: int`` retains the int64
default. Quote the whole annotation or use ``from __future__ import annotations``
to defer eager Python evaluation of header symbols. Captured runtime ``typing.TypeVar``
objects are not script symbols; ordinary Python typing uses remain unaffected.

An explicit scalar annotation ``n: n`` preserves a captured native symbol's identity.
An independently typed parameter such as ``n: T.int32`` and ordinary body locals
shadow definition captures normally. Annotation classes, Python unions and deferred
return-constructor evaluation retain their builder behavior.

Syntax and construction protocol
--------------------------------

The syntax transpiler rewrites a fresh Python AST using scope and declaration facts
from a prescan. A registered decorator selects the construction namespace. Assignments
call binding hooks, expression statements call emission hooks, and control flow opens
builder frames. The namespace owns the meaning of these operations and the supported
IR constructs.

``tvm.script.parser.protocol_registry`` records syntax policies under registered
namespace paths. These policies identify scalar annotations, mutable declarations and
result span handling. Symbolic shapes use concrete expressions. Source aliases resolve to those paths;
ordinary Python calls remain calls in the generated program. Explicit ``constexpr``
markers select host control flow during construction.

Generated operations retain source locations. Syntax restrictions raise source-located
``SyntaxError`` exceptions; builder and Python helper errors retain their original
exception types. Temporary parse state is released when construction finishes or fails.

Frame-based construction
------------------------

The IR builder maintains an active stack of frames, such as module, function and loop
frames. Entering a frame establishes its scope; exiting finalizes its IR and attaches it
to its parent. Frames own parameters, symbols and construction state. Builder hooks own
value binding, type rules and validation of completed IR. The shared
``resolve_global_info_args`` decorator resolves named metadata arguments after
ordinary Python argument evaluation; dialect callbacks own selector syntax and
lookup in the active module.

Public script namespaces expose decorators and construction operations. The same
underlying builders can also be used directly from Python. The transpiler therefore
needs no separate mutable IR representation: it generates calls to this construction
protocol.

Printing and round trips
------------------------

``IRDocsifier`` dispatches on IR types and context to produce a ``Doc`` tree of
expressions and statements. It tracks scopes and names; ``DocToPythonScript`` formats
the tree as Python text. This document tree is separate from the parser's Python AST.

For example, a small function can be authored, printed and parsed again:

.. code-block:: python

   import tvm
   from tvm.script import tirx as T

   @T.prim_func
   def increment(A: T.Buffer((4,), "float32")):
       for i in T.serial(4):
           A[i] = A[i] + 1.0

   text = increment.script()
   reparsed = tvm.script.from_source(text)
   tvm.ir.assert_structural_equal(increment, reparsed)

Printed source uses canonical forms rather than preserving the original spelling.
Round trips require printable IR and any external objects needed by the source;
the text does not serialize arbitrary Python state.

Namespace extension
-------------------

A language variant supplies construction hooks and a public script namespace, then
registers its aliases with ``register_namespace``. ``register_namespace_initializer``
supports lazy setup. ``tvm.script.register_dialect`` exposes a package through the
public script namespace. Syntax policies are registered beside the operations that
need them, while IR printing is extended through docsifier registrations.

This division keeps source acquisition, syntax translation and formatting shared.
Language-specific construction and validation remain with the namespace and its
builders.
