.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements. See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership. The ASF licenses this file
   to you under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

CUDA compilation configuration
==============================

.. autoclass:: tvm.backend.cuda.CompileConfig
   :members:
   :undoc-members:

.. autoclass:: tvm.backend.cuda.compiler.CompilationResult
   :members:

.. autofunction:: tvm.backend.cuda.compiler.compile_source

The dataclass signature above is generated directly from the field registry used
for validation and compiler-flag translation. Every field defaults to ``None``
(unspecified) at the API boundary. Resolution fills compiler defaults only after
entry overrides have been applied. See :doc:`../native_basics/cuda/compiling`
for precedence, defaults, artifacts and migration examples.

See :doc:`cuda_compile_options` for the NVCC/NVRTC/ptxas option inventory,
current coverage, version differences, and known validation gaps.
