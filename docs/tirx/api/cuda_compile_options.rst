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

``CompileConfig`` currently exposes a selected set of CUDA compilation controls.
The three raw-option sequences provide additional arguments; accepting an argument
does not establish that its compilation phase, artifact type, or compiler version
is supported by TIRx.

Audit baseline
--------------

This inventory was checked on 2026-10-10 against the CUDA 13.2 manuals,
NVCC 13.2.51, NVRTC 13.2, and the installed CUDA 13.2 ``ptxas --help``.
The current implementation was inspected in ``backend/cuda/compile_config.py``,
``backend/cuda/compiler.py``, and ``support/nvcc.py``.

The :download:`complete option inventory <cuda_compile_options.json>` contains
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

Current structured fields
-------------------------

.. list-table:: Mapping to compiler controls
   :header-rows: 1
   :widths: 24 38 38

   * - Field
     - NVCC / NVRTC route
     - Current boundary
   * - ``arch``
     - GPU architecture
     - One real ``sm_*`` architecture per compilation group. NVRTC PTX output
       translates it to the corresponding ``compute_*`` spelling.
   * - ``compiler``
     - Select the frontend
     - ``nvcc`` or ``nvrtc``; default is NVRTC.
   * - ``target_format``
     - NVCC phase flag / NVRTC output API
     - PTX, cubin, or NVCC fatbin. No general object, LTO IR, or OptiX IR path.
   * - ``fast_math``
     - ``--use_fast_math``
     - Defaults to true in TIRx. Individual math fields follow the preset.
   * - ``ftz``, ``prec_div``, ``prec_sqrt``, ``fmad``
     - Corresponding frontend flags
     - Explicit true/false values override the preset.
   * - ``cxx_standard``
     - ``--std``
     - C++11, C++14, C++17, C++20. No toolchain-version capability check.
   * - ``device_debug``, ``lineinfo``
     - Device debug and line-info flags
     - Both default to false. Device debug can affect optimization as well as
       debug information; there is no structured ``dopt`` field.
   * - ``ptxas_opt_level``
     - ptxas ``--opt-level`` via frontend forwarding
     - Integer 0 through 3. This is not NVCC host optimization.
   * - ``ptxas_reg_usage_level``
     - ptxas ``--register-usage-level``
     - Integer 0 through 10; TIRx defaults to 10. This is an optimization
       heuristic, not a register-count cap.
   * - ``include_dirs``, ``defines``
     - Include search paths and macro definitions
     - Tuples replace inherited tuples. Driver-required CUDA includes are added
       separately.
   * - ``nvcc_options``, ``nvrtc_options``, ``ptxas_options``
     - Backend-specific forwarding
     - No comprehensive option schema, version validation, or workflow validation.
   * - ``dump_dir``
     - TIRx source, artifact, metadata, and log output
     - Diagnostics only; excluded from the compilation cache identity.

Configuration inheritance applies to these fields, not to the contents of raw
argument lists. For example, an entry's ``nvrtc_options`` replaces the inherited
sequence. A raw sequence containing flags for an inactive frontend is currently
retained in metadata but is not used by the selected frontend.

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
     - Mostly raw arguments. Add explicit validation of frontend availability,
       accepted values, and interactions with debug settings.
   * - Compilation time and determinism
     - ``--split-compile``, NVCC ``--threads`` and
       ``--split-compile-extended``, ``--frandom-seed``
     - Raw arguments. Treat host build parallelism separately from GPU launch
       dimensions. Extended split compilation requires its own workflow checks.
   * - Register and occupancy constraints
     - ``--maxrregcount``; ptxas ``--device-function-maxrregcount``,
       ``--maxntid``, ``--minnctapersm``, ``--override-directive-values``
     - Need precedence rules with ``KernelAttributes`` and ``LaunchConfig``.
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
       lifetime, and cache dependencies are not represented by CompileConfig.
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

The driver still inserts settings outside the dataclass field registry:

* NVCC receives ``-O3``. This controls its host optimization level, while
  ``ptxas_opt_level`` is forwarded to the GPU assembler.
* NVRTC receives default-device execution-space and device-int128 flags, CUDA
  header search paths, and ``--no-cache`` on NVRTC 12.9 or later. The cache flag
  avoids implicit CUDA-driver initialization and an observed CUDA 13.2 cubin
  cache collision across FTZ settings.
* Both frontends receive ptxas verbosity and local-memory-use warnings.
* NVSHMEM compilation adds its include paths, relocatable-code settings, and
  link inputs. Output is constrained to cubin.

The source/options/toolchain identity currently includes configuration values
and the reported frontend version. It does not hash included-header contents,
PCH files, or arbitrary external libraries named by raw arguments. An option
inventory must not be mistaken for a fully reproducible external-input model.

Confirmed validation gaps
-------------------------

These are observations of the current implementation, not supported alternatives:

* The shared prefix rule rejects ``-Ofc`` although its long spelling
  ``--Ofast-compile`` is accepted. It also blocks NVCC ``-O`` while allowing
  ``--optimize`` to reach a command that already contains ``-O3``.
* ptxas ``-regUsageLevel`` is accepted although ``--register-usage-level`` is
  reserved for ``ptxas_reg_usage_level``. Similarly, ``--gpu-name`` is accepted
  while its ``-arch`` alias is blocked. Alias normalization must be stage-specific.
* ``--maxrregcount`` is rejected as though it duplicates a CompileConfig field,
  but there is no such field. KernelAttributes provides a declaration-level
  alternative, with different scope. The diagnostic and precedence policy need
  to state that distinction.
* Several phase-changing, entry-filtering, and forwarded linker arguments are
  accepted without the corresponding artifact workflow. Unknown raw arguments
  are delegated to the compiler, so accepted does not mean supported.
* Structured fields have type/range checks but no general compiler-version or
  architecture capability matrix. C++23 cannot be selected even with a compiler
  that supports it, and version-dependent raw options are checked only by the tool.

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

Follow-up implementation order
------------------------------

1. Give each registered option a frontend/stage, canonical name, exact aliases,
   value type, scope, known version requirements, and interaction rules. Generate
   documentation and raw-option ownership checks from that metadata.
2. Fix the confirmed alias/ownership holes and misleading diagnostics. Explicitly
   reject unsupported phase changes. Test both long and short spellings.
3. Add structured controls for common device optimization, compilation-time,
   diagnostics, and preprocessing settings. Retain an explicit backend-specific
   escape hatch for options outside the portable subset.
4. Decide the semantics of compilation-wide resource defaults separately from
   per-kernel declaration attributes, including how they apply to shared helpers.
5. Add RDC/LTO, new artifact types, PCH management, and independent host-build
   configuration only together with their required compile/link/load workflows.

This audit adds documentation and an inventory. It does not claim that these
follow-up changes or every option in the inventory are implemented.

.. _NVCC 13.2 manual: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html
.. _NVRTC 13.2 manual: https://docs.nvidia.com/cuda/archive/13.2.0/nvrtc/index.html
.. _NVCC 13.4 manual: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
.. _NVRTC 13.4 manual: https://docs.nvidia.com/cuda/nvrtc/index.html
