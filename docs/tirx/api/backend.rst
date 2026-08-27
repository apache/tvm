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

These exports are loaded lazily, so they need explicit autodoc entries:

.. autoclass:: tvm.backend.cuda.lang.BaseTileScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.ClusterPersistentScheduler2D
   :members:

.. autoclass:: tvm.backend.cuda.lang.FlashAttentionLPTScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.FlashAttentionLinearScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.GroupMajor3D
   :members:

.. autoclass:: tvm.backend.cuda.lang.IndexedTripleTileScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.MBarrier
   :members:

.. autoclass:: tvm.backend.cuda.lang.Pipeline
   :members:

.. autoclass:: tvm.backend.cuda.lang.PipelineState
   :members:

.. autoclass:: tvm.backend.cuda.lang.RankAwareGroupMajorTileScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.SMEMPool
   :members:

.. autoclass:: tvm.backend.cuda.lang.SmemDescriptor
   :members:

.. autoclass:: tvm.backend.cuda.lang.TCGen05Bar
   :members:

.. autoclass:: tvm.backend.cuda.lang.TMABar
   :members:

.. autoclass:: tvm.backend.cuda.lang.TMEMPool
   :members:

.. autoclass:: tvm.backend.cuda.lang.WarpRole
   :members:

.. autoclass:: tvm.backend.cuda.lang.WarpgroupRole
   :members:

tvm.backend.cuda.op
*******************
.. automodule:: tvm.backend.cuda.op
   :members:
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
   :undoc-members:
   :no-index:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.cpp
********************
.. automodule:: tvm.backend.cuda.cpp
   :members:
   :imported-members:
   :no-index:
   :exclude-members: PrimExpr, Op, Call

tvm.backend.cuda.iket
*********************
.. automodule:: tvm.backend.cuda.iket
   :members:
   :no-index:

tvm.backend.cuda.ptx
********************
The table-driven direct PTX namespace and its extension API are documented on
:doc:`ptx`; they are not repeated here.

tvm.backend.cuda.tile_primitive
*******************************
This package imports CUDA dispatch implementations for their registration side
effects.  The stable extension API is documented under
:ref:`tirx-api-dispatch-extension-points`; individual dispatches are described
in :doc:`../tile_primitives`.

tvm.backend.cuda.target_tags
****************************
.. automodule:: tvm.backend.cuda.target_tags

Importing this module registers NVIDIA target strings such as
``nvidia/nvidia-h100`` and ``nvidia/nvidia-b100``.  Its helper functions are
private; the imported generic ``tvm.target.register_tag`` function is not part
of this module's API.

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
   :undoc-members:
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

tvm.backend.trn.target_tags
***************************
.. automodule:: tvm.backend.trn.target_tags
   :no-index:

Importing this module registers ``aws/trn1/trn1.2xlarge`` and
``aws/trn1/trn1.32xlarge``.  The imported generic
``tvm.target.register_tag`` function is not part of this module's API.

tvm.backend.trn.tile_primitive
******************************
This package imports Trainium dispatch implementations for registration side
effects.  It does not define a separate public API; use the common dispatch
extension points documented in :doc:`script`.

tvm.backend.trn.transform
*************************
.. automodule:: tvm.backend.trn.transform
   :members:
   :imported-members:
   :no-index:

The two lazily exported pass classes require explicit autodoc entries:

.. autoclass:: tvm.backend.trn.transform.TrnNaiveAllocator
   :members:
   :no-index:

.. autoclass:: tvm.backend.trn.transform.TrnPrivateBufferAlloc
   :members:
   :no-index:
