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

tvm.backend.cuda
================

The CUDA backend — the tile-primitive dispatch, intrinsic builders, the ``T.cuda``
/ ``T.ptx`` script namespaces, and the shared/tensor-memory pools — lives under
``tvm.backend.cuda``, separate from the TIRx frontend (``tvm.tirx``). Other
backends sit alongside it (``tvm.backend.rocm`` and so on).

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
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.script
***********************
.. automodule:: tvm.backend.cuda.script
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.operator
*************************
.. automodule:: tvm.backend.cuda.operator
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.target_tags
****************************
.. automodule:: tvm.backend.cuda.target_tags
   :members:
   :imported-members:
   :exclude-members: PrimExpr, Op, Call
