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

tvm.script.ir_builder
---------------------

tvm.script.ir_builder
*********************
.. automodule:: tvm.script.ir_builder
   :members:
   :imported-members:
   :exclude-members: GenericConst, Range, StringImm, StringType

tvm.script.ir_builder.parser_protocol
*************************************
Shared module construction and location helpers are exported directly from
:mod:`tvm.script.ir_builder`. The protocol below describes the hooks implemented
by each language variant, including :func:`~tvm.script.ir_builder.parser_protocol.if_`
and :func:`~tvm.script.ir_builder.parser_protocol.function_`.

.. automodule:: tvm.script.ir_builder.parser_protocol
   :members:

Dialect builders are owned by their respective script packages. Registered
compatibility paths under ``tvm.script.ir_builder`` resolve to the same modules.

tvm.relax.script.ir_builder
***************************
.. automodule:: tvm.relax.script.ir_builder
   :members:
   :exclude-members: ExternFunc, ShapeExpr, TupleGetItem, Range

tvm.relax.script.ir_builder.distributed
***************************************
.. automodule:: tvm.relax.script.ir_builder.distributed
   :members:

tvm.tirx.script.ir_builder
**************************
.. automodule:: tvm.tirx.script.ir_builder
   :members:
   :exclude-members: Var, Call, CommReducer, Reduce, SMEMPool, TMEMPool, FloatImm, IntImm, Cast, Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, Min, Max, EQ, NE, LT, LE, GT, GE, And, Or, Not, Select, Ramp, Broadcast, Shuffle, CallEffectKind, IterVar

Re-exported IR nodes retain the APIs documented in :mod:`tvm.ir` and
:mod:`tvm.tirx` and :mod:`tvm.te`; CUDA allocation pools are documented with their backend.
These aliases remain available from the builder namespace:

.. autosummary::

   tvm.ir.Var
   tvm.ir.Call
   tvm.te.CommReducer
   tvm.te.Reduce
   tvm.backend.cuda.lang.alloc_pool.SMEMPool
   tvm.backend.cuda.lang.alloc_pool.TMEMPool
   tvm.tirx.FloatImm
   tvm.tirx.IntImm
   tvm.tirx.Cast
   tvm.tirx.Add
   tvm.tirx.Sub
   tvm.tirx.Mul
   tvm.tirx.Div
   tvm.tirx.Mod
   tvm.tirx.FloorDiv
   tvm.tirx.FloorMod
   tvm.tirx.LShift
   tvm.tirx.RShift
   tvm.tirx.BitwiseAnd
   tvm.tirx.BitwiseOr
   tvm.tirx.BitwiseXor
   tvm.tirx.BitwiseNot
   tvm.tirx.Min
   tvm.tirx.Max
   tvm.tirx.EQ
   tvm.tirx.NE
   tvm.tirx.LT
   tvm.tirx.LE
   tvm.tirx.GT
   tvm.tirx.GE
   tvm.tirx.And
   tvm.tirx.Or
   tvm.tirx.Not
   tvm.tirx.Select
   tvm.tirx.Ramp
   tvm.tirx.Broadcast
   tvm.tirx.Shuffle
   tvm.tirx.CallEffectKind
   tvm.tirx.IterVar
