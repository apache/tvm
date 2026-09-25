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

tvm.script
----------

tvm.script
**********
.. automodule:: tvm.script
   :members:
   :imported-members:
   :exclude-members: Any

tvm.script.relax
****************
.. automodule:: tvm.script.relax
   :members:
   :exclude-members: ExternFunc, ShapeExpr, TupleGetItem, Range, function, macro

.. autofunction:: tvm.script.relax.function

.. autofunction:: tvm.script.relax.macro

tvm.script.tirx
***************
.. automodule:: tvm.script.tirx
   :members:
   :exclude-members: Range, meta_var, Var, Call, CommReducer, Reduce, SMEMPool, TMEMPool, FloatImm, IntImm, Cast, Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, Min, Max, EQ, NE, LT, LE, GT, GE, And, Or, Not, Select, Ramp, Broadcast, Shuffle, CallEffectKind, IterVar, ComposeLayout, DtypeConstructor, ExecScope, Iter, Layout, LetAnnotation, LocalVectorAnnotation, ScopeIdDef, TileLayout, Buffer, buffer, prim_func, jit, inline, macro

.. autofunction:: tvm.script.tirx.prim_func

.. autofunction:: tvm.script.tirx.jit

.. autofunction:: tvm.script.tirx.inline

.. autofunction:: tvm.script.tirx.macro
