.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements. See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership. The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

CUDA compiler option coverage
=============================

CUDA uses a fixed set of configuration keys and native toolchain argument lists.
This inventory is documentation, not a registry of accepted flags. Compiler
upgrades do not require updating the Python API or this inventory before new
options can be used.

Audit baseline
--------------

This inventory was checked on 2026-10-10 against the CUDA 13.2 manuals,
NVCC 13.2.51, NVRTC 13.2, and the installed CUDA 13.2 ``ptxas --help``.
The current implementation was inspected in ``backend/cuda/backend_config.py``,
``backend/cuda/compiler.py``, and ``support/nvcc.py``.

The :download:`complete option inventory <../../_static/tirx/cuda_compile_options.json>` contains
147 NVCC options, 59 NVRTC options, and 57 ptxas options. Each JSON entry records long
and short spellings, its intended category, the current API route, and whether
the current raw-option validator accepts each spelling. These are option names,
not all possible values or every historical alias. NVCC's inventory combines its
142 documented main-driver options with five additional names in local help.
Options belonging to nvlink itself are outside this inventory; the NVCC forwarding
option is included.

The ``raw_validation`` field probes argument ownership only. It does not run
NVCC, NVRTC, or ptxas and does not certify a value, platform, or combination.
The inventory is a snapshot to review when changing the validator or supported
toolkit baseline; it is not a second runtime registry.

Sources: `NVCC 13.2 manual`_, `NVRTC 13.2 manual`_, and local tool help. The
`NVCC 13.4 manual`_ and `NVRTC 13.4 manual`_ were compared separately below.

API routes
----------

``arch``, ``compiler``, and ``target_format`` select the architecture, frontend,
and artifact. Other options go directly into the selected toolchain list:

.. code-block:: python

    backend_config = {"cuda": {
        "arch": "sm_100a",
        "compiler": "nvcc",
        "nvcc": ["--use_fast_math", "--ftz=false", "--std=c++20", "-I/include"],
        "nvrtc": ["--use_fast_math", "--ftz=false", "--std=c++20", "-I/include"],
        "ptxas": ["-O3", "--register-usage-level=10"],
    }}

These lists are argv items, without shell parsing. The selected frontend receives
its own list and the forwarded ``ptxas`` list; the other frontend list is retained
but unused. Each list replaces the inherited list completely. For example,
``nvrtc=[]`` removes the default fast-math flag and ``ptxas=[]`` removes the
default assembler arguments. See :doc:`cuda_compile` for all six fixed keys.

TVM checks key names and types and reserves a small set of architecture, output,
and tool-routing flags. General optimization, math, preprocessing, debug, and
assembler flags are passed through. Their versions, values, and combinations
are validated by the actual compiler, not a TVM option schema.

Controls requiring separate treatment
-------------------------------------

.. list-table:: Additional option families
   :header-rows: 1
   :widths: 23 37 40

   * - Family
     - Examples
     - TIRx coverage and required policy
   * - Device optimization
     - ``--dopt``, ``--Ofast-compile``,
       ``--extra-device-vectorization``, ``--jump-table-density``
     - Native arguments; the selected compiler validates availability, accepted
       values, and interactions with debug settings.
   * - Compilation time and determinism
     - ``--split-compile``, NVCC ``--threads`` and
       ``--split-compile-extended``, ``--frandom-seed``
     - Raw arguments. Treat host build parallelism separately from GPU launch
       dimensions. Extended split compilation requires its own workflow checks.
   * - Register and occupancy constraints
     - ``--maxrregcount``; ptxas ``--device-function-maxrregcount``,
       ``--maxntid``, ``--minnctapersm``, ``--override-directive-values``
     - Native arguments; CUDA defines interaction with declaration attributes.
       A compilation-wide register cap also concerns device helpers; it is not
       equivalent to the kernel-only ``__maxnreg__`` declaration attribute.
   * - Memory code generation
     - ptxas ``--def-load-cache``, ``--force-load-cache``,
       ``--def-store-cache``, ``--force-store-cache``
     - Raw ptxas arguments. Do not conflate them with layout, memory scope,
       or instruction-specific cache qualifiers.
   * - Language and preprocessing
     - ``--undefine-macro``, ``--pre-include``, ``--restrict``;
       NVCC relaxed constexpr / extended lambda flags;
       NVRTC builtin type and execution-space controls
     - Mostly raw arguments. ``restrict`` is a caller assertion about aliasing,
       not a harmless optimization preset. Header and precompiled-header contents
       need an external-dependency policy if used in persistent caches.
   * - Diagnostics and instrumentation
     - ``--diag-error``, ``--diag-warn``, ``--diag-suppress``, warning controls,
       optimization reports, timing, stack protection, sanitizer controls
     - Mostly raw arguments. Distinguish log-only options from options that change
       generated instructions. Warning categories differ between frontends.
   * - NVRTC precompiled headers
     - ``--pch``, ``--create-pch``, ``--use-pch``, ``--pch-dir`` and related flags
     - Raw acceptance is not artifact management. PCH files, compatibility,
       lifetime, and cache dependencies are not represented by BackendConfig.
   * - Relocatable code and device linking
     - ``--device-c``, ``--device-w``, ``--relocatable-device-code``,
       ``--extensible-whole-program``, ``--device-link``
     - No general user-supplied device-library/link-input API. NVSHMEM has a
       dedicated internal linking path; that does not implement arbitrary RDC.
   * - LTO and alternative outputs
     - ``--dlink-time-opt``, ``--gen-opt-lto``, NVCC ``--ltoir``, ``--optix-ir``
     - Need output retrieval, linking, serialization, and loading support.
       Raw acceptance alone cannot provide these artifact pipelines.
   * - Multiple architectures
     - NVCC ``--generate-code`` and ``--gpu-code``
     - Intentionally blocked. Separate device entries can use separate
       architectures; one compilation group has one architecture.
   * - Host compiler and linker
     - NVCC ``--compiler-bindir``, ``--compiler-options``, ``--linker-options``,
       ``--cudart``, host linker-script controls
     - Some affect NVCC's host-toolchain stages. They do not configure the
       independent C++ build used to export ``cuda_host``. A host-build API is
       a separate concern from per-device-entry compiler settings.
   * - Driver utilities and phase control
     - Help/version, dependency generation, preprocessing-only, execution,
       output paths, options files, entry selection
     - Keep query/build-system actions out of a kernel compilation API.
       Reject flags that change the output contract or hide effective arguments.

Driver settings that are currently implicit
-------------------------------------------

The driver inserts integration settings separately from the user argument lists:

* NVCC receives ``-O3``. This controls its host optimization level, while
  ``ptxas=["-O..."]`` controls the GPU assembler.
* NVRTC receives default-device execution-space and device-int128 flags, CUDA
  header search paths, and ``--no-cache`` on NVRTC 12.9 or later. The cache flag
  avoids implicit CUDA-driver initialization and an observed CUDA 13.2 cubin
  cache collision across FTZ settings.
* The default ``ptxas`` list enables verbosity and local-memory-use warnings;
  overriding the list replaces these defaults.
* NVSHMEM compilation adds its include paths, relocatable-code settings, and
  link inputs. Output is constrained to cubin.

Scope and limitations
---------------------

The configuration snapshot records option strings. It does not hash included
headers, PCH files, or external libraries. Process-local factory caches assume
those inputs and the installed toolchain stay unchanged for the cache lifetime.
Source artifacts replay the saved options using the toolchain available at load
time; they do not bundle a compiler.

Accepting an argument does not add a new artifact or linking workflow. General
RDC/LTO linking, PCH lifecycle management, alternative output formats, and host
compiler configuration need their own integration. Only PTX, cubin, and NVCC
fatbin are supported by this compilation path.

Version comparison
------------------

The baseline above is CUDA 13.2, matching the tools used for validation. Comparing
13.2 and 13.4 documentation also identifies the following changes. This comparison
does not establish the exact release in which each option first appeared.

* Both newer manuals add C++23 to the language choices, conversion-warning and
  error-limit controls, UTF-8 source handling, and CUDA Tile frontend controls.
* NVCC adds Tile artifact phases/forwarding, pruning/concatenation controls, and
  ``--apply-controls``. Their appearance in NVCC does not make them TIRx artifact
  formats or TIRx tile-operation APIs.
* NVRTC adds default-Tile/implicit-Tile-variable controls and bundled-header
  selection. These are frontend-specific options, not portable common fields.

Verify an option against the deployed compiler, output format, and architecture;
do not accept it merely because it appears in the newest manual.

Maintaining this inventory
--------------------------

Update the documentary snapshot when reviewing a toolkit upgrade. Do not add
per-option runtime fields or generate a validator from this inventory. New
artifact or linking workflows require explicit adapter and runtime support;
ordinary compiler flags can already be passed through the toolchain lists.

.. _NVCC 13.2 manual: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html
.. _NVRTC 13.2 manual: https://docs.nvidia.com/cuda/archive/13.2.0/nvrtc/index.html
.. _NVCC 13.4 manual: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
.. _NVRTC 13.4 manual: https://docs.nvidia.com/cuda/nvrtc/index.html
