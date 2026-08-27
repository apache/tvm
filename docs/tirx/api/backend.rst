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

Backend APIs
============

The CUDA backend — the tile-primitive dispatch, intrinsic builders, the ``T.cuda``
/ ``T.ptx`` script namespaces, and the shared/tensor-memory pools — lives under
``tvm.backend.cuda``, separate from the TIRx frontend (``tvm.tirx``). Trainium
support follows the same ownership model under ``tvm.backend.trn``.  Load hooks
in ``tvm.backend`` register a backend's script namespaces, compilation pipeline,
target tags, and tile-primitive implementations.  See :doc:`ptx` for the
user-facing direct PTX instruction surface and how it differs from ``T.cuda``.

tvm.backend
***********
.. automodule:: tvm.backend
   :members:
   :imported-members:
   :no-index:

tvm.backend.cuda
****************
.. automodule:: tvm.backend.cuda
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.lang
*********************
.. automodule:: tvm.backend.cuda.lang
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.op
*******************
.. automodule:: tvm.backend.cuda.op
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call, PrimType, PointerType

tvm.backend.cuda.script
***********************
.. automodule:: tvm.backend.cuda.script
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.codegen
************************
.. automodule:: tvm.backend.cuda.codegen
   :members:
   :imported-members:
   :no-index:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.cpp
********************
.. automodule:: tvm.backend.cuda.cpp
   :members:
   :imported-members:
   :no-index:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.ptx
********************
.. automodule:: tvm.backend.cuda.ptx
   :members:
   :imported-members:
   :no-index:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.tile_primitive
*******************************
.. automodule:: tvm.backend.cuda.tile_primitive
   :members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.target_tags
****************************
.. automodule:: tvm.backend.cuda.target_tags
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.transforms
***************************
.. automodule:: tvm.backend.cuda.transforms
   :members:
   :imported-members:
   :no-index:

tvm.backend.trn
***************
.. automodule:: tvm.backend.trn
   :members:
   :imported-members:
   :no-index:

tvm.backend.trn.layout
**********************
.. automodule:: tvm.backend.trn.layout
   :members:
   :no-index:

tvm.backend.trn.op
******************
.. automodule:: tvm.backend.trn.op
   :members:
   :no-index:

tvm.backend.trn.pipeline
************************
.. automodule:: tvm.backend.trn.pipeline
   :members:
   :no-index:

tvm.backend.trn.script
**********************
.. automodule:: tvm.backend.trn.script
   :members:
   :no-index:

tvm.backend.trn.tile_primitive
******************************
.. automodule:: tvm.backend.trn.tile_primitive
   :members:
   :imported-members:
   :no-index:

tvm.backend.trn.transform
*************************
.. automodule:: tvm.backend.trn.transform
   :members:
   :imported-members:
   :no-index:
