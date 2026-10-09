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

Asynchronous tensor transfers
=============================

The instruction name fixes the transfer mechanism. Every interface issues the
transfer; its caller supplies the corresponding completion protocol.

* :doc:`copy_async/ldgsts`: ``Tx.cuda.tile.cp_async`` copies global to shared.
  The caller commits and waits with the ``cp.async`` group operations.
* :doc:`copy_async/tma`: ``cp_async_bulk_tensor_load``,
  ``cp_async_bulk_tensor_store``, and ``cp_reduce_async_bulk_tensor`` use TMA.
  Loads signal an mbarrier; stores and reductions use bulk async groups.
* :doc:`copy_async/dsmem`: ``cp_async_bulk`` copies between CTAs' shared memory
  and signals the destination CTA's mbarrier.
* :doc:`copy_async/tcgen05_cp`: ``tcgen05.cp`` copies shared memory to TMEM.
* :doc:`copy_async/tcgen05_ldst`: ``tcgen05.ld`` and ``tcgen05.st`` transfer
  between TMEM and registers at warpgroup scope.

All names above are in ``Tx.cuda.tile``. Tensor-memory transfers require the
matching ``tcgen05`` completion operations.

.. toctree::
   :maxdepth: 1

   copy_async/ldgsts
   copy_async/tma
   copy_async/dsmem
   copy_async/tcgen05_cp
   copy_async/tcgen05_ldst
