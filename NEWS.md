<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

TVM Change Log
==============

  - [On-going version](#on-going-version)
  - [0.8](#08)
    - [Accepted RFCs](#accepted-rfcs)
    - [Features and Improvements](#features-and-improvements)
      - [TE, TIR, TVMScript](#te-tir-tvmscript)
      - [AutoTVM, AutoScheduler, Meta Schedule](#autotvm-autoscheduler-meta-schedule)
      - [Operator Coverage](#operator-coverage)
      - [Training](#training)
      - [Relay](#relay)
      - [MicroTVM, AOT, Graph Executor and VM](#microtvm-aot-graph-executor-and-vm)
      - [Arithmetic Analysis](#arithmetic-analysis)
      - [Frontends](#frontends)
      - [Codegen Backends and Runtime](#codegen-backends-and-runtime)
      - [BYOC Integration with Vendor Libraries: TensorRT, ACL, VitisAI](#byoc-integration-with-vendor-libraries-tensorrt-acl-vitisai)
      - [TVMC](#tvmc)
      - [Rust Binding](#rust-binding)
      - [Misc](#misc)
  - [0.7](#07)
    - [New Features](#new-features)
      - [Automatic Scheduling (Experimental)](#automatic-scheduling-experimental)
      - [BYOC](#byoc)
      - [Operator Coverage](#operator-coverage-1)
      - [Quantization](#quantization)
      - [Relay](#relay-1)
      - [Runtime and Backend](#runtime-and-backend)
      - [Rust Support](#rust-support)
      - [TIR](#tir)
      - [TE](#te)
      - [TVMC(Experimental)](#tvmcexperimental)
    - [Feature Improvement](#feature-improvement)
      - [Accelerator and Microcontroller Support](#accelerator-and-microcontroller-support)
      - [Arithmetic Analysis](#arithmetic-analysis-1)
      - [AutoTVM and Graph Tuner](#autotvm-and-graph-tuner)
      - [BYOC](#byoc-1)
      - [Codegen](#codegen)
      - [Dynamism Support](#dynamism-support)
      - [Frontend and User Interface](#frontend-and-user-interface)
      - [Relay](#relay-2)
      - [Operator Coverage](#operator-coverage-2)
      - [Runtime and Backend](#runtime-and-backend-1)
      - [Quantization](#quantization-1)
      - [TE](#te-1)
      - [TIR](#tir-1)
      - [Performance Improvements](#performance-improvements)
      - [Documentation](#documentation)
      - [Bug Fixes](#bug-fixes)
    - [API Changes](#api-changes)
    - [Deprecation](#deprecation)
  - [0.6](#06)
    - [Relay in Production](#relay-in-production)
    - [Relay Virtual Machine](#relay-virtual-machine)
    - [Training](#training-1)
    - [Quantization](#quantization-2)
    - [Accelerator and Microcontroller Support](#accelerator-and-microcontroller-support-1)
    - [Rust Support](#rust-support-1)
    - [Operator Support](#operator-support)
    - [Frontend and User Interface](#frontend-and-user-interface-1)
    - [Runtime and Backend Support](#runtime-and-backend-support)
    - [Language and Architecture](#language-and-architecture)
    - [Symbolic shape enhancement](#symbolic-shape-enhancement)
    - [Language and Architecture](#language-and-architecture-1)
    - [Arithmetic Analysis](#arithmetic-analysis-2)
    - [Runtime and Backend Support](#runtime-and-backend-support-1)
    - [Frontend and User Interface](#frontend-and-user-interface-2)
    - [AutoTVM](#autotvm)
    - [Performance Improvements](#performance-improvements-1)
    - [Documentation](#documentation-1)
    - [Build and Test](#build-and-test)
    - [Bug Fixes](#bug-fixes-1)
    - [Known Issues](#known-issues)
    - [Depreciations](#depreciations)
  - [0.5](#05)
  - [0.4](#04)
  - [0.3](#03)
  - [0.2](#02)
  - [0.1](#01)
  - [Initial version](#initial-version)

This file records the changes in TVM library in reverse chronological order.

## On-going version

Refer to the Roadmap issue for complete list on on-going version features.
If you check in something that is not reflected in Roadmap issue, please reply
to that issue so it can get added.

## 0.8

Apache TVM v0.8 brings several major exciting experimental features, including:
- PaddlePaddle frontend
- TVMScript: round-trippable python-based syntax for TIR
- TorchScript integration
- TensorIR scheduling language
- TensorRT and CUTLASS integration via BYOC
- Int4 TensorCore support in AutoTVM
- MicroTVM Project API and Zephyr, Arduino support
- AOT executor
- Robost Windows support
- Affine analysis infra: iter-affine-map
- Improved Vulkan backend
- CUDA graph support in TVM runtime

Besides, The community has been working together to refactor and evolve the existing infrastructure, including but not limited to:
- Relay compilation engine
- Relay pattern language
- CI and build process
- Refactoring documentation and tutorials
- Stablizing AutoScheduler
- Stablizing TVMC command line driver interface
- Stablizing target system
- Frontend coverage, quantization, dynamic shape, training

Full changelog: https://gist.github.com/junrushao1994/c669905dbc41edc2e691316df49d8562.

### Accepted RFCs

The community has adopted a [formal RFC process](https://github.com/apache/tvm-rfcs). Below is a list of the formal RFCs accepted by the community since then:
- [RFC-0005] Meta schedule (AutoTIR)
- [RFC-0006] Automatic mixed-precision pass and support
- [RFC-0007] Parametrized unit tests
- [RFC-0008] MicroTVM Project API
- [RFC-0009] Unified static memory planner
- [RFC-0010] Target-registered compiler flow customisation
- [RFC-0011] Arm® Ethos-U integration
- [RFC-0014] Pipeline executor
- [RFC-0015] Use CMSIS-NN with TVM
- [RFC-0019] Add PaddlePaddle frontend
- [RFC-0020] Extend metadata in project option
- [RFC-0022] TIR non-scalar constants
- [RFC-0023] Adding annotation field to `tir.allocate` nodes
- [RFC-0025] PyTorchTVM
- [RFC-0027] Formalize TVM documentation organization
- [RFC-0028] Command line composition from internal registry
- [RFC-0029] Migrating target attributes to IRModule
- [RFC-0030] Command line configuration files
- [RFC-0031] C Device API
- [RFC-0036] TVMScript namespace
- [RFC-0041] Update TVMScript block syntax

### Features and Improvements
#### TE, TIR, TVMScript

- TVMScript parser and printer [#7630](https://github.com/apache/tvm/pull/7630) [#9115](https://github.com/apache/tvm/pull/9115) [#9286](https://github.com/apache/tvm/pull/9286)
- Scheduleable TIR (S-TIR) infrastructure, analysis and lowering passes [#7553](https://github.com/apache/tvm/pull/7553) [#7765](https://github.com/apache/tvm/pull/7765) [#7847](https://github.com/apache/tvm/pull/7847) [#8114](https://github.com/apache/tvm/pull/8114) [#8121](https://github.com/apache/tvm/pull/8121) [#7873](https://github.com/apache/tvm/pull/7873) [#7923](https://github.com/apache/tvm/pull/7923) [#7962](https://github.com/apache/tvm/pull/7962) [#7848](https://github.com/apache/tvm/pull/7848) [#8044](https://github.com/apache/tvm/pull/8044) [#7806](https://github.com/apache/tvm/pull/7806)
- S-TIR schedule primitives: `compute-inline`, `reverse-compute-inline`, `fuse`, `split`, `rfactor`, `storage-align`, `vectorize`, `unroll`, `bind`, `reorder`, `cache-read`, `cache-write`, `compute-at`, `reverse-compute-at`, `decompose-reduction` [#8170](https://github.com/apache/tvm/pull/8170) [#8467](https://github.com/apache/tvm/pull/8467) [#8544](https://github.com/apache/tvm/pull/8544) [#8693](https://github.com/apache/tvm/pull/8693) [#8716](https://github.com/apache/tvm/pull/8716) [#8767](https://github.com/apache/tvm/pull/8767) [#8863](https://github.com/apache/tvm/pull/8863) [#8943](https://github.com/apache/tvm/pull/8943) [#9041](https://github.com/apache/tvm/pull/9041)
- While loop in TIR [#7425](https://github.com/apache/tvm/pull/7425) [#9004](https://github.com/apache/tvm/pull/9004)
- Metaprogramming in S-TIR via `specialize` [#8354](https://github.com/apache/tvm/pull/8354)
- Support Return value in TIR [#7084](https://github.com/apache/tvm/pull/7084) [#7932](https://github.com/apache/tvm/pull/7932)
- Storage scope support in `PointerType` [#8017](https://github.com/apache/tvm/pull/8017) [#8366](https://github.com/apache/tvm/pull/8366) [#8463](https://github.com/apache/tvm/pull/8463)
- Creation of S-TIR via TE compute [#7987](https://github.com/apache/tvm/pull/7987)

#### AutoTVM, AutoScheduler, Meta Schedule

- PopenPoolExecutor is used to replace python native library to provide better multiprocessing support as well as enable auto-tuning in Jupyter notebooks for AutoTVM and AutoScheduler [#6959](https://github.com/apache/tvm/pull/6959) [#8492](https://github.com/apache/tvm/pull/8492) [#8913](https://github.com/apache/tvm/pull/8913) [#8820](https://github.com/apache/tvm/pull/8820) [#8851](https://github.com/apache/tvm/pull/8851)
- AutoScheduler improvement and stabilization: task scheduler, layout rewrite, early stopping, dispatching [#6945](https://github.com/apache/tvm/pull/6945) [#6750](https://github.com/apache/tvm/pull/6750) [#6987](https://github.com/apache/tvm/pull/6987) [#7156](https://github.com/apache/tvm/pull/7156) [#8862](https://github.com/apache/tvm/pull/8862) [#8995](https://github.com/apache/tvm/pull/8995) [#7571](https://github.com/apache/tvm/pull/7571) [#7376](https://github.com/apache/tvm/pull/7376) [#7377](https://github.com/apache/tvm/pull/7377) [#7344](https://github.com/apache/tvm/pull/7344) [#7185](https://github.com/apache/tvm/pull/7185)
- AutoScheduler support for sparse workloads [#7313](https://github.com/apache/tvm/pull/7313) [#7635](https://github.com/apache/tvm/pull/7635) [#8065](https://github.com/apache/tvm/pull/8065)
- AutoScheduler support for Vulkan, ROCm, Mali [#7626](https://github.com/apache/tvm/pull/7626) [#7038](https://github.com/apache/tvm/pull/7038) [#7132](https://github.com/apache/tvm/pull/7132)
- AutoTVM support for int4 TensorCore [#7831](https://github.com/apache/tvm/pull/7831) [#8402](https://github.com/apache/tvm/pull/8402)
- Meta Schedule core infrastructure, builder runner and database [#8615](https://github.com/apache/tvm/pull/8615) [#8623](https://github.com/apache/tvm/pull/8623) [#8642](https://github.com/apache/tvm/pull/8642) [#8817](https://github.com/apache/tvm/pull/8817) [#9079](https://github.com/apache/tvm/pull/9079) [#9132](https://github.com/apache/tvm/pull/9132) [#9154](https://github.com/apache/tvm/pull/9154) [#9053](https://github.com/apache/tvm/pull/9053) [#9059](https://github.com/apache/tvm/pull/9059) [#9044](https://github.com/apache/tvm/pull/9044) [#9111](https://github.com/apache/tvm/pull/9111) [#9061](https://github.com/apache/tvm/pull/9061) [#9153](https://github.com/apache/tvm/pull/9153)

#### Operator Coverage
- Operators for Int-8 vision transformer on GPU [#7814](https://github.com/apache/tvm/pull/7814)
- Optimizing NMS and ROI-related kernel on GPU [#7257](https://github.com/apache/tvm/pull/7257) [#7172](https://github.com/apache/tvm/pull/7172) [#7136](https://github.com/apache/tvm/pull/7136) [#7796](https://github.com/apache/tvm/pull/7796) [#7463](https://github.com/apache/tvm/pull/7463) [#6516](https://github.com/apache/tvm/pull/6516) [#7440](https://github.com/apache/tvm/pull/7440) [#7666](https://github.com/apache/tvm/pull/7666) [#8174](https://github.com/apache/tvm/pull/8174)
- Support and optimize sparse operators [#8605](https://github.com/apache/tvm/pull/8605) [#7477](https://github.com/apache/tvm/pull/7477) [#7435](https://github.com/apache/tvm/pull/7435) [#6889](https://github.com/apache/tvm/pull/6889) [#6580](https://github.com/apache/tvm/pull/6580) [#8437](https://github.com/apache/tvm/pull/8437)
- Sort-related operators and optimization [#9184](https://github.com/apache/tvm/pull/9184) [#7669](https://github.com/apache/tvm/pull/7669) [#8672](https://github.com/apache/tvm/pull/8672) [#7611](https://github.com/apache/tvm/pull/7611) [#7195](https://github.com/apache/tvm/pull/7195) [#7056](https://github.com/apache/tvm/pull/7056) [#6978](https://github.com/apache/tvm/pull/6978)
- Support for einsum operator [#6370](https://github.com/apache/tvm/pull/6370)
- Matmul, dense operators and their optimization [#8921](https://github.com/apache/tvm/pull/8921) [#8527](https://github.com/apache/tvm/pull/8527) [#8234](https://github.com/apache/tvm/pull/8234) [#8250](https://github.com/apache/tvm/pull/8250) [#6616](https://github.com/apache/tvm/pull/6616) [#8229](https://github.com/apache/tvm/pull/8229) [#8401](https://github.com/apache/tvm/pull/8401) [#7404](https://github.com/apache/tvm/pull/7404) [#8669](https://github.com/apache/tvm/pull/8669)
- Convolution and pooling operators and their optimization [#8620](https://github.com/apache/tvm/pull/8620) [#8936](https://github.com/apache/tvm/pull/8936) [#8584](https://github.com/apache/tvm/pull/8584) [#7075](https://github.com/apache/tvm/pull/7075) [#7142](https://github.com/apache/tvm/pull/7142) [#7515](https://github.com/apache/tvm/pull/7515) [#6999](https://github.com/apache/tvm/pull/6999) [#6899](https://github.com/apache/tvm/pull/6899) [#6840](https://github.com/apache/tvm/pull/6840) [#6137](https://github.com/apache/tvm/pull/6137) [#6802](https://github.com/apache/tvm/pull/6802) [#6445](https://github.com/apache/tvm/pull/6445) [#6711](https://github.com/apache/tvm/pull/6711) [#6714](https://github.com/apache/tvm/pull/6714) [#8167](https://github.com/apache/tvm/pull/8167) [#8222](https://github.com/apache/tvm/pull/8222) [#8275](https://github.com/apache/tvm/pull/8275) [#8276](https://github.com/apache/tvm/pull/8276) [#8422](https://github.com/apache/tvm/pull/8422) [#8430](https://github.com/apache/tvm/pull/8430) [#6687](https://github.com/apache/tvm/pull/6687) [#7928](https://github.com/apache/tvm/pull/7928) [#8897](https://github.com/apache/tvm/pull/8897)
- Scatter and gather operators and their optimization [#8479](https://github.com/apache/tvm/pull/8479) [#7600](https://github.com/apache/tvm/pull/7600) [#7044](https://github.com/apache/tvm/pull/7044) [#7464](https://github.com/apache/tvm/pull/7464) [#7233](https://github.com/apache/tvm/pull/7233) [#6533](https://github.com/apache/tvm/pull/6533) [#6856](https://github.com/apache/tvm/pull/6856) [#6854](https://github.com/apache/tvm/pull/6854) [#7927](https://github.com/apache/tvm/pull/7927) [#8105](https://github.com/apache/tvm/pull/8105)
- Prefix scan, cumsum and cumprod [#7722](https://github.com/apache/tvm/pull/7722) [#7303](https://github.com/apache/tvm/pull/7303) [#7314](https://github.com/apache/tvm/pull/7314) [#7334](https://github.com/apache/tvm/pull/7334) [#7123](https://github.com/apache/tvm/pull/7123) [#6868](https://github.com/apache/tvm/pull/6868)
- Dynamic shape and shape functions [#7414](https://github.com/apache/tvm/pull/7414) [#6979](https://github.com/apache/tvm/pull/6979) [#6912](https://github.com/apache/tvm/pull/6912) [#6898](https://github.com/apache/tvm/pull/6898) [#6373](https://github.com/apache/tvm/pull/6373) [#8068](https://github.com/apache/tvm/pull/8068) [#7490](https://github.com/apache/tvm/pull/7490) [#7487](https://github.com/apache/tvm/pull/7487)
- Miscellaneous improvement. Operators including: reshape, resize, pad, PRNG, transpose, where, softmax, concat, nll_loss, space_to_batch_nd, batch_to_space_nd, slice_like; Libraries including thrust, cuDNN, cuBLAS, MIOpen; Improving schedules for generic reduction and softmax. [#8592](https://github.com/apache/tvm/pull/8592) [#7375](https://github.com/apache/tvm/pull/7375) [#7287](https://github.com/apache/tvm/pull/7287) [#7184](https://github.com/apache/tvm/pull/7184) [#7131](https://github.com/apache/tvm/pull/7131) [#7086](https://github.com/apache/tvm/pull/7086) [#7083](https://github.com/apache/tvm/pull/7083) [#8030](https://github.com/apache/tvm/pull/8030) [#6851](https://github.com/apache/tvm/pull/6851) [#6477](https://github.com/apache/tvm/pull/6477) [#8346](https://github.com/apache/tvm/pull/8346) [#6759](https://github.com/apache/tvm/pull/6759) [#8028](https://github.com/apache/tvm/pull/8028) [#8056](https://github.com/apache/tvm/pull/8056) [#8369](https://github.com/apache/tvm/pull/8369) [#7468](https://github.com/apache/tvm/pull/7468) [#7458](https://github.com/apache/tvm/pull/7458) [#7194](https://github.com/apache/tvm/pull/7194) [#8138](https://github.com/apache/tvm/pull/8138) [#8543](https://github.com/apache/tvm/pull/8543)

#### Training

- Relay AutoDiff [#7677](https://github.com/apache/tvm/pull/7677) [#8318](https://github.com/apache/tvm/pull/8318)
- TE AutoDiff [#7321](https://github.com/apache/tvm/pull/7321)
- Gradient operators [#7685](https://github.com/apache/tvm/pull/7685) [#7340](https://github.com/apache/tvm/pull/7340) [#6767](https://github.com/apache/tvm/pull/6767) [#8307](https://github.com/apache/tvm/pull/8307) [#7357](https://github.com/apache/tvm/pull/7357) [#6827](https://github.com/apache/tvm/pull/6827)

#### Relay

- Pattern language and mixed-mode visitor: matching more IR constructs, fuzzy matching; converting more passes to non-recursive.  [#8843](https://github.com/apache/tvm/pull/8843) [#7754](https://github.com/apache/tvm/pull/7754) [#7355](https://github.com/apache/tvm/pull/7355) [#7332](https://github.com/apache/tvm/pull/7332) [#7282](https://github.com/apache/tvm/pull/7282) [#7151](https://github.com/apache/tvm/pull/7151) [#7120](https://github.com/apache/tvm/pull/7120) [#6958](https://github.com/apache/tvm/pull/6958) [#7507](https://github.com/apache/tvm/pull/7507) [#8325](https://github.com/apache/tvm/pull/8325) [#8774](https://github.com/apache/tvm/pull/8774) [#7817](https://github.com/apache/tvm/pull/7817) [#7374](https://github.com/apache/tvm/pull/7374) [#6695](https://github.com/apache/tvm/pull/6695) [#6704](https://github.com/apache/tvm/pull/6704)
- Improving or adding passes including ExtractOperators, SimplifyExpr, DynamicToStatic, DefuseOps, ConvertLayout, FoldConstant. Added a set of utilities that allows a model to be run efficiently on TensorCores [#9253](https://github.com/apache/tvm/pull/9253) [#9245](https://github.com/apache/tvm/pull/9245) [#8996](https://github.com/apache/tvm/pull/8996) [#7827](https://github.com/apache/tvm/pull/7827) [#9034](https://github.com/apache/tvm/pull/9034) [#7807](https://github.com/apache/tvm/pull/7807) [#8755](https://github.com/apache/tvm/pull/8755) [#7731](https://github.com/apache/tvm/pull/7731) [#7368](https://github.com/apache/tvm/pull/7368) [#7603](https://github.com/apache/tvm/pull/7603) [#7656](https://github.com/apache/tvm/pull/7656) [#7423](https://github.com/apache/tvm/pull/7423) [#7354](https://github.com/apache/tvm/pull/7354) [#6946](https://github.com/apache/tvm/pull/6946) [#6748](https://github.com/apache/tvm/pull/6748) [#6720](https://github.com/apache/tvm/pull/6720) [#6776](https://github.com/apache/tvm/pull/6776) [#7835](https://github.com/apache/tvm/pull/7835) [#7895](https://github.com/apache/tvm/pull/7895) [#8205](https://github.com/apache/tvm/pull/8205)
- TECompiler and refactoring of compilation workflow [#9103](https://github.com/apache/tvm/pull/9103) [#8974](https://github.com/apache/tvm/pull/8974) [#8886](https://github.com/apache/tvm/pull/8886) [#8802](https://github.com/apache/tvm/pull/8802) [#8501](https://github.com/apache/tvm/pull/8501) [#8526](https://github.com/apache/tvm/pull/8526) [#8486](https://github.com/apache/tvm/pull/8486) [#8597](https://github.com/apache/tvm/pull/8597) [#7518](https://github.com/apache/tvm/pull/7518) [#7552](https://github.com/apache/tvm/pull/7552) [#8914](https://github.com/apache/tvm/pull/8914) [#9130](https://github.com/apache/tvm/pull/9130)
- Quantization and automatic-mixed precision [#8883](https://github.com/apache/tvm/pull/8883) [#8810](https://github.com/apache/tvm/pull/8810) [#8644](https://github.com/apache/tvm/pull/8644) [#7613](https://github.com/apache/tvm/pull/7613) [#8069](https://github.com/apache/tvm/pull/8069) [#8341](https://github.com/apache/tvm/pull/8341) [#8126](https://github.com/apache/tvm/pull/8126) [#8460](https://github.com/apache/tvm/pull/8460)
- Parser, printer and diagnostic [#7347](https://github.com/apache/tvm/pull/7347) [#6274](https://github.com/apache/tvm/pull/6274) [#6692](https://github.com/apache/tvm/pull/6692) [#8352](https://github.com/apache/tvm/pull/8352) [#8000](https://github.com/apache/tvm/pull/8000)

#### MicroTVM, AOT, Graph Executor and VM

- Pipeline Executor [#8702](https://github.com/apache/tvm/pull/8702) [#9108](https://github.com/apache/tvm/pull/9108)
- CUDA graph integration in graph executor [#7616](https://github.com/apache/tvm/pull/7616)
- Enable add `set_output_zero_copy` in graph executor [#8497](https://github.com/apache/tvm/pull/8497)
- VM: memory allocation improvement, shape function improvement and misc [#7746](https://github.com/apache/tvm/pull/7746) [#7451](https://github.com/apache/tvm/pull/7451) [#7413](https://github.com/apache/tvm/pull/7413) [#7210](https://github.com/apache/tvm/pull/7210) [#8040](https://github.com/apache/tvm/pull/8040) [#6938](https://github.com/apache/tvm/pull/6938) [#8661](https://github.com/apache/tvm/pull/8661) [#7676](https://github.com/apache/tvm/pull/7676) [#8285](https://github.com/apache/tvm/pull/8285)
- AOT compilation and execution [#8697](https://github.com/apache/tvm/pull/8697) [#7785](https://github.com/apache/tvm/pull/7785) [#8014](https://github.com/apache/tvm/pull/8014) [#8023](https://github.com/apache/tvm/pull/8023) [#8096](https://github.com/apache/tvm/pull/8096) [#8075](https://github.com/apache/tvm/pull/8075)
- Project API infrastructure: [#8380](https://github.com/apache/tvm/pull/8380) [#8963](https://github.com/apache/tvm/pull/8963) [#8708](https://github.com/apache/tvm/pull/8708) [#8019](https://github.com/apache/tvm/pull/8019)
- MicroTVM, Zephyr, Arduino RVM, AutoTVM support [#9320](https://github.com/apache/tvm/pull/9320) [#8941](https://github.com/apache/tvm/pull/8941) [#7804](https://github.com/apache/tvm/pull/7804) [#7786](https://github.com/apache/tvm/pull/7786) [#7449](https://github.com/apache/tvm/pull/7449) [#7891](https://github.com/apache/tvm/pull/7891) [#7915](https://github.com/apache/tvm/pull/7915) [#8055](https://github.com/apache/tvm/pull/8055) [#8037](https://github.com/apache/tvm/pull/8037) [#8386](https://github.com/apache/tvm/pull/8386) [#8519](https://github.com/apache/tvm/pull/8519) [#8748](https://github.com/apache/tvm/pull/8748) [8154](https://github.com/apache/tvm/pull/8154) [#8945](https://github.com/apache/tvm/pull/8945) [#8624](https://github.com/apache/tvm/pull/8624) [#8701](https://github.com/apache/tvm/pull/8701) [#7723](https://github.com/apache/tvm/pull/7723) [#8715](https://github.com/apache/tvm/pull/8715) [#7225](https://github.com/apache/tvm/pull/7225) [#6964](https://github.com/apache/tvm/pull/6964) [#7813](https://github.com/apache/tvm/pull/7813) [#7528](https://github.com/apache/tvm/pull/7528)
- The pure C runtime (CRT) [#7398](https://github.com/apache/tvm/pull/7398) [#7333](https://github.com/apache/tvm/pull/7333) [#7095](https://github.com/apache/tvm/pull/7095) [#7225](https://github.com/apache/tvm/pull/7225)
- Model library format [#8270](https://github.com/apache/tvm/pull/8270) [#8072](https://github.com/apache/tvm/pull/8072) [#7938](https://github.com/apache/tvm/pull/7938)

#### Arithmetic Analysis

- Tighter bounds and more simplification on cast [#6771](https://github.com/apache/tvm/pull/6771) [#7045](https://github.com/apache/tvm/pull/7045)
- Introducing iterator (quasi-) affine map detection [#6667](https://github.com/apache/tvm/pull/6667) [#7752](https://github.com/apache/tvm/pull/7752) [#7759](https://github.com/apache/tvm/pull/7759)
- Inverse of iterator affine map [#8384](https://github.com/apache/tvm/pull/8384) [#8427](https://github.com/apache/tvm/pull/8427)
- Subspace division in iterator affine map [#7760](https://github.com/apache/tvm/pull/7760)

#### Frontends

- PaddlePaddle initial support [#8645](https://github.com/apache/tvm/pull/8645)  [#9124](https://github.com/apache/tvm/pull/9124) [#9126](https://github.com/apache/tvm/pull/9126) [#9295](https://github.com/apache/tvm/pull/9295) [#9370](https://github.com/apache/tvm/pull/9370) [#9236](https://github.com/apache/tvm/pull/9236) [#9283](https://github.com/apache/tvm/pull/9283)
- ONNX support, including better handling of control flow, coverage of more operators, better dynamic shape support, more tests. [#9265](https://github.com/apache/tvm/pull/9265) [#9178](https://github.com/apache/tvm/pull/9178) [#9146](https://github.com/apache/tvm/pull/9146) [#8894](https://github.com/apache/tvm/pull/8894) [#8966](https://github.com/apache/tvm/pull/8966) [#8967](https://github.com/apache/tvm/pull/8967) [#7818](https://github.com/apache/tvm/pull/7818) [#9000](https://github.com/apache/tvm/pull/9000) [#9001](https://github.com/apache/tvm/pull/9001) [#9066](https://github.com/apache/tvm/pull/9066) [#9028](https://github.com/apache/tvm/pull/9028) [#9002](https://github.com/apache/tvm/pull/9002) [#8985](https://github.com/apache/tvm/pull/8985) [#9019](https://github.com/apache/tvm/pull/9019) [#9017](https://github.com/apache/tvm/pull/9017) [#8972](https://github.com/apache/tvm/pull/8972) [#7802](https://github.com/apache/tvm/pull/7802) [#7800](https://github.com/apache/tvm/pull/7800) [#7781](https://github.com/apache/tvm/pull/7781) [#8919](https://github.com/apache/tvm/pull/8919) [#9054](https://github.com/apache/tvm/pull/9054) [#8906](https://github.com/apache/tvm/pull/8906) [#8933](https://github.com/apache/tvm/pull/8933) [#8959](https://github.com/apache/tvm/pull/8959) [#8907](https://github.com/apache/tvm/pull/8907) [#7771](https://github.com/apache/tvm/pull/7771) [#8923](https://github.com/apache/tvm/pull/8923) [#8924](https://github.com/apache/tvm/pull/8924) [#7755](https://github.com/apache/tvm/pull/7755) [#7720](https://github.com/apache/tvm/pull/7720) [#8773](https://github.com/apache/tvm/pull/8773) [#8872](https://github.com/apache/tvm/pull/8872) [#7655](https://github.com/apache/tvm/pull/7655) [#8741](https://github.com/apache/tvm/pull/8741) [#7633](https://github.com/apache/tvm/pull/7633) [#8781](https://github.com/apache/tvm/pull/8781) [#8866](https://github.com/apache/tvm/pull/8866) [#8867](https://github.com/apache/tvm/pull/8867) [#7522](https://github.com/apache/tvm/pull/7522) [#7519](https://github.com/apache/tvm/pull/7519) [#7489](https://github.com/apache/tvm/pull/7489) [#7438](https://github.com/apache/tvm/pull/7438) [#7429](https://github.com/apache/tvm/pull/7429) [#7364](https://github.com/apache/tvm/pull/7364) [#7300](https://github.com/apache/tvm/pull/7300) [#7259](https://github.com/apache/tvm/pull/7259) [#7243](https://github.com/apache/tvm/pull/7243) [#7237](https://github.com/apache/tvm/pull/7237) [#7208](https://github.com/apache/tvm/pull/7208) [#7189](https://github.com/apache/tvm/pull/7189) [#7115](https://github.com/apache/tvm/pull/7115) [#7109](https://github.com/apache/tvm/pull/7109) [#7089](https://github.com/apache/tvm/pull/7089) [#7036](https://github.com/apache/tvm/pull/7036) [#7031](https://github.com/apache/tvm/pull/7031) [#6839](https://github.com/apache/tvm/pull/6839) [#6351](https://github.com/apache/tvm/pull/6351) [#7842](https://github.com/apache/tvm/pull/7842) [#7844](https://github.com/apache/tvm/pull/7844) [#6646](https://github.com/apache/tvm/pull/6646) [#6647](https://github.com/apache/tvm/pull/6647) [#6681](https://github.com/apache/tvm/pull/6681) [#6700](https://github.com/apache/tvm/pull/6700) [#7883](https://github.com/apache/tvm/pull/7883) [#6726](https://github.com/apache/tvm/pull/6726) [#6730](https://github.com/apache/tvm/pull/6730) [#7899](https://github.com/apache/tvm/pull/7899) [#7900](https://github.com/apache/tvm/pull/7900) [#7906](https://github.com/apache/tvm/pull/7906) [#7934](https://github.com/apache/tvm/pull/7934) [#7956](https://github.com/apache/tvm/pull/7956) [#8007](https://github.com/apache/tvm/pull/8007) [#8011](https://github.com/apache/tvm/pull/8011) [#8084](https://github.com/apache/tvm/pull/8084) [#8099](https://github.com/apache/tvm/pull/8099) [#8189](https://github.com/apache/tvm/pull/8189) [#8191](https://github.com/apache/tvm/pull/8191) [#8304](https://github.com/apache/tvm/pull/8304) [#8321](https://github.com/apache/tvm/pull/8321) [#8337](https://github.com/apache/tvm/pull/8337) [#8356](https://github.com/apache/tvm/pull/8356) [#8385](https://github.com/apache/tvm/pull/8385) [#8502](https://github.com/apache/tvm/pull/8502) [#8426](https://github.com/apache/tvm/pull/8426) [#8440](https://github.com/apache/tvm/pull/8440) [#8456](https://github.com/apache/tvm/pull/8456) [#8475](https://github.com/apache/tvm/pull/8475) [#7391](https://github.com/apache/tvm/pull/7391) [#7394](https://github.com/apache/tvm/pull/7394) [#8621](https://github.com/apache/tvm/pull/8621) [#8322](https://github.com/apache/tvm/pull/8322) [#8323](https://github.com/apache/tvm/pull/8323) [#8435](https://github.com/apache/tvm/pull/8435) [#8436](https://github.com/apache/tvm/pull/8436) [#8455](https://github.com/apache/tvm/pull/8455) [#7353](https://github.com/apache/tvm/pull/7353) [#7215](https://github.com/apache/tvm/pull/7215)
- TensorFlow and TFLite, including more operators, better TensorArray support and quantization [#9404](https://github.com/apache/tvm/pull/9404) [#9256](https://github.com/apache/tvm/pull/9256) [#8689](https://github.com/apache/tvm/pull/8689) [#7789](https://github.com/apache/tvm/pull/7789) [#7736](https://github.com/apache/tvm/pull/7736) [#8763](https://github.com/apache/tvm/pull/8763) [#8647](https://github.com/apache/tvm/pull/8647) [#8648](https://github.com/apache/tvm/pull/8648) [#8558](https://github.com/apache/tvm/pull/8558) [#8780](https://github.com/apache/tvm/pull/8780) [#8538](https://github.com/apache/tvm/pull/8538) [#7659](https://github.com/apache/tvm/pull/7659) [#7639](https://github.com/apache/tvm/pull/7639) [#7531](https://github.com/apache/tvm/pull/7531) [#7520](https://github.com/apache/tvm/pull/7520) [#7502](https://github.com/apache/tvm/pull/7502) [#7496](https://github.com/apache/tvm/pull/7496) [#7473](https://github.com/apache/tvm/pull/7473) [#7452](https://github.com/apache/tvm/pull/7452) [#7442](https://github.com/apache/tvm/pull/7442) [#7441](https://github.com/apache/tvm/pull/7441) [#7400](https://github.com/apache/tvm/pull/7400) [#7320](https://github.com/apache/tvm/pull/7320) [#7293](https://github.com/apache/tvm/pull/7293) [#7267](https://github.com/apache/tvm/pull/7267) [#7159](https://github.com/apache/tvm/pull/7159) [#7148](https://github.com/apache/tvm/pull/7148) [#7114](https://github.com/apache/tvm/pull/7114) [#7113](https://github.com/apache/tvm/pull/7113) [#7093](https://github.com/apache/tvm/pull/7093) [#7074](https://github.com/apache/tvm/pull/7074) [#7048](https://github.com/apache/tvm/pull/7048) [#7030](https://github.com/apache/tvm/pull/7030) [#6998](https://github.com/apache/tvm/pull/6998) [#6984](https://github.com/apache/tvm/pull/6984) [#6970](https://github.com/apache/tvm/pull/6970) [#6949](https://github.com/apache/tvm/pull/6949) [#6933](https://github.com/apache/tvm/pull/6933) [#6918](https://github.com/apache/tvm/pull/6918) [#6901](https://github.com/apache/tvm/pull/6901) [#6885](https://github.com/apache/tvm/pull/6885) [#6849](https://github.com/apache/tvm/pull/6849) [#5767](https://github.com/apache/tvm/pull/5767) [#6589](https://github.com/apache/tvm/pull/6589) [#6670](https://github.com/apache/tvm/pull/6670) [#6674](https://github.com/apache/tvm/pull/6674) [#6675](https://github.com/apache/tvm/pull/6675) [#7866](https://github.com/apache/tvm/pull/7866) [#6685](https://github.com/apache/tvm/pull/6685) [#7885](https://github.com/apache/tvm/pull/7885) [#6729](https://github.com/apache/tvm/pull/6729) [#7901](https://github.com/apache/tvm/pull/7901) [#6774](https://github.com/apache/tvm/pull/6774) [#6783](https://github.com/apache/tvm/pull/6783) [#6799](https://github.com/apache/tvm/pull/6799) [#7951](https://github.com/apache/tvm/pull/7951) [#8024](https://github.com/apache/tvm/pull/8024) [#8051](https://github.com/apache/tvm/pull/8051) [#8060](https://github.com/apache/tvm/pull/8060) [#8074](https://github.com/apache/tvm/pull/8074) [#8142](https://github.com/apache/tvm/pull/8142) [#8179](https://github.com/apache/tvm/pull/8179) [#8251](https://github.com/apache/tvm/pull/8251) [#8277](https://github.com/apache/tvm/pull/8277) [#8335](https://github.com/apache/tvm/pull/8335) [#8364](https://github.com/apache/tvm/pull/8364) [#8375](https://github.com/apache/tvm/pull/8375) [#8431](https://github.com/apache/tvm/pull/8431) [#8454](https://github.com/apache/tvm/pull/8454) [#6818](https://github.com/apache/tvm/pull/6818) [#8483](https://github.com/apache/tvm/pull/8483) [#9099](https://github.com/apache/tvm/pull/9099) [#9165](https://github.com/apache/tvm/pull/9165)
- PyTorch: more operators including activations, inplace operators, RNNs, NMS [#9371](https://github.com/apache/tvm/pull/9371) [#9204](https://github.com/apache/tvm/pull/9204) [#9185](https://github.com/apache/tvm/pull/9185) [#9135](https://github.com/apache/tvm/pull/9135) [#9133](https://github.com/apache/tvm/pull/9133) [#9015](https://github.com/apache/tvm/pull/9015) [#8839](https://github.com/apache/tvm/pull/8839) [#8718](https://github.com/apache/tvm/pull/8718) [#8699](https://github.com/apache/tvm/pull/8699) [#8692](https://github.com/apache/tvm/pull/8692) [#7712](https://github.com/apache/tvm/pull/7712) [#8753](https://github.com/apache/tvm/pull/8753) [#7694](https://github.com/apache/tvm/pull/7694) [#8583](https://github.com/apache/tvm/pull/8583) [#7675](https://github.com/apache/tvm/pull/7675) [#7646](https://github.com/apache/tvm/pull/7646) [#7606](https://github.com/apache/tvm/pull/7606) [#7592](https://github.com/apache/tvm/pull/7592) [#7569](https://github.com/apache/tvm/pull/7569) [#7544](https://github.com/apache/tvm/pull/7544) [#7549](https://github.com/apache/tvm/pull/7549) [#7535](https://github.com/apache/tvm/pull/7535) [#7517](https://github.com/apache/tvm/pull/7517) [#7465](https://github.com/apache/tvm/pull/7465) [#7397](https://github.com/apache/tvm/pull/7397) [#7371](https://github.com/apache/tvm/pull/7371) [#7348](https://github.com/apache/tvm/pull/7348) [#7346](https://github.com/apache/tvm/pull/7346) [#7325](https://github.com/apache/tvm/pull/7325) [#7231](https://github.com/apache/tvm/pull/7231) [#7174](https://github.com/apache/tvm/pull/7174) [#7154](https://github.com/apache/tvm/pull/7154) [#7137](https://github.com/apache/tvm/pull/7137) [#7134](https://github.com/apache/tvm/pull/7134) [#7133](https://github.com/apache/tvm/pull/7133) [#7128](https://github.com/apache/tvm/pull/7128) [#7088](https://github.com/apache/tvm/pull/7088) [#7023](https://github.com/apache/tvm/pull/7023) [#6900](https://github.com/apache/tvm/pull/6900) [#6602](https://github.com/apache/tvm/pull/6602) [#7845](https://github.com/apache/tvm/pull/7845) [#6659](https://github.com/apache/tvm/pull/6659) [#6740](https://github.com/apache/tvm/pull/6740) [#6782](https://github.com/apache/tvm/pull/6782) [#6784](https://github.com/apache/tvm/pull/6784) [#7958](https://github.com/apache/tvm/pull/7958) [#8192](https://github.com/apache/tvm/pull/8192) [#8397](https://github.com/apache/tvm/pull/8397) [#8398](https://github.com/apache/tvm/pull/8398) [#8403](https://github.com/apache/tvm/pull/8403) [#8447](https://github.com/apache/tvm/pull/8447) [#6829](https://github.com/apache/tvm/pull/6829)
- MXNet support. More operators and NLP model coverage in GluonNLP [#7568](https://github.com/apache/tvm/pull/7568) [#7409](https://github.com/apache/tvm/pull/7409) [#7209](https://github.com/apache/tvm/pull/7209) [#7191](https://github.com/apache/tvm/pull/7191) [#7062](https://github.com/apache/tvm/pull/7062) [#6561](https://github.com/apache/tvm/pull/6561) [#6699](https://github.com/apache/tvm/pull/6699)
- Misc: CoreML, Keras, DarkNet, etc. [#7667](https://github.com/apache/tvm/pull/7667) [#6676](https://github.com/apache/tvm/pull/6676) [#6651](https://github.com/apache/tvm/pull/6651) [#6963](https://github.com/apache/tvm/pull/6963) [#7949](https://github.com/apache/tvm/pull/7949) [#7035](https://github.com/apache/tvm/pull/7035) [#7446](https://github.com/apache/tvm/pull/7446) [#8562](https://github.com/apache/tvm/pull/8562) [#8599](https://github.com/apache/tvm/pull/8599)

#### Codegen Backends and Runtime

- LLVM backend: recover LLVM support on windows; support target feature strings in function attributes; atomic support in NVPTX, ROCm; LLVM compatibility to LLVM 12+ [#9305](https://github.com/apache/tvm/pull/9305) [#9223](https://github.com/apache/tvm/pull/9223) [#9138](https://github.com/apache/tvm/pull/9138) [#8860](https://github.com/apache/tvm/pull/8860) [#8958](https://github.com/apache/tvm/pull/8958) [#6763](https://github.com/apache/tvm/pull/6763) [#6698](https://github.com/apache/tvm/pull/6698) [#6717](https://github.com/apache/tvm/pull/6717) [#6738](https://github.com/apache/tvm/pull/6738) [#8293](https://github.com/apache/tvm/pull/8293) [#6907](https://github.com/apache/tvm/pull/6907) [#7051](https://github.com/apache/tvm/pull/7051)
- ROCm 3.9 bitcode files search [#6865](https://github.com/apache/tvm/pull/6865)
- Vulkan and SPIR-V refactoring and major improvement in codegen and runtime. [A critical bug fix in SPIRV codegen](https://github.com/apache/tvm/pull/8102) allows the Vulkan backend to produce correct outputs on more hardwares and drivers. Added support for querying device specific hardware parameters and capabilities, dynamic shapes, irregular ops such as sorting and NMS, UBO, fp16, and vectorization. We can now run complicated models like MaskRCNN on Vulkan end to end. [#8904](https://github.com/apache/tvm/pull/8904) [#7833](https://github.com/apache/tvm/pull/7833) [#7717](https://github.com/apache/tvm/pull/7717) [#7681](https://github.com/apache/tvm/pull/7681) [#8746](https://github.com/apache/tvm/pull/8746) [#8813](https://github.com/apache/tvm/pull/8813) [#7609](https://github.com/apache/tvm/pull/7609) [#8882](https://github.com/apache/tvm/pull/8882) [#7607](https://github.com/apache/tvm/pull/7607) [#7591](https://github.com/apache/tvm/pull/7591) [#7574](https://github.com/apache/tvm/pull/7574) [#7572](https://github.com/apache/tvm/pull/7572) [#7833](https://github.com/apache/tvm/pull/7833) [#6662](https://github.com/apache/tvm/pull/6662) [#7969](https://github.com/apache/tvm/pull/7969) [#8013](https://github.com/apache/tvm/pull/8013) [#8048](https://github.com/apache/tvm/pull/8048) [#8098](https://github.com/apache/tvm/pull/8098) [#8102](https://github.com/apache/tvm/pull/8102) [#8107](https://github.com/apache/tvm/pull/8107) [#8127](https://github.com/apache/tvm/pull/8127) [#8151](https://github.com/apache/tvm/pull/8151) [#8196](https://github.com/apache/tvm/pull/8196) [#8320](https://github.com/apache/tvm/pull/8320) [#8588](https://github.com/apache/tvm/pull/8588) [#8332](https://github.com/apache/tvm/pull/8332) [#8333](https://github.com/apache/tvm/pull/8333) [#8348](https://github.com/apache/tvm/pull/8348) [#8528](https://github.com/apache/tvm/pull/8528)
- Metal language version upgrade (`MTLLanguageVersion2_3`), better codegen support, int64 support, various bug fixes [#7830](https://github.com/apache/tvm/pull/7830) [#7819](https://github.com/apache/tvm/pull/7819) [#7714](https://github.com/apache/tvm/pull/7714) [#7118](https://github.com/apache/tvm/pull/7118) [#7116](https://github.com/apache/tvm/pull/7116) [#7105](https://github.com/apache/tvm/pull/7105) [#7980](https://github.com/apache/tvm/pull/7980) [#8054](https://github.com/apache/tvm/pull/8054) [#8175](https://github.com/apache/tvm/pull/8175) [#8202](https://github.com/apache/tvm/pull/8202) [#8206](https://github.com/apache/tvm/pull/8206) [#8313](https://github.com/apache/tvm/pull/8313)
- OpenCL, VTA, Verilator: refactored code generator, better error messages, various bug fixes [#7834](https://github.com/apache/tvm/pull/7834) [#7777](https://github.com/apache/tvm/pull/7777) [#7761](https://github.com/apache/tvm/pull/7761) [#7100](https://github.com/apache/tvm/pull/7100) [#6125](https://github.com/apache/tvm/pull/6125) [#6126](https://github.com/apache/tvm/pull/6126) [#6191](https://github.com/apache/tvm/pull/6191) [#7834](https://github.com/apache/tvm/pull/7834) [#8256](https://github.com/apache/tvm/pull/8256) [#8257](https://github.com/apache/tvm/pull/8257) [#8731](https://github.com/apache/tvm/pull/8731) [#8756](https://github.com/apache/tvm/pull/8756) [#8973](https://github.com/apache/tvm/pull/8973)
- CUDA: enable `__launch_bounds__`, dynamic shared memory, TensorCore, BF16, half2, NVCC version upgrade [#9341](https://github.com/apache/tvm/pull/9341) [#8678](https://github.com/apache/tvm/pull/8678) [#7561](https://github.com/apache/tvm/pull/7561) [#7273](https://github.com/apache/tvm/pull/7273) [#7146](https://github.com/apache/tvm/pull/7146) [#7147](https://github.com/apache/tvm/pull/7147) [#7099](https://github.com/apache/tvm/pull/7099) [#7065](https://github.com/apache/tvm/pull/7065) [#7033](https://github.com/apache/tvm/pull/7033) [#7014](https://github.com/apache/tvm/pull/7014) [#7907](https://github.com/apache/tvm/pull/7907) [#7964](https://github.com/apache/tvm/pull/7964) [#9087](https://github.com/apache/tvm/pull/9087) [#8135](https://github.com/apache/tvm/pull/8135) [#8137](https://github.com/apache/tvm/pull/8137) [#8457](https://github.com/apache/tvm/pull/8457) [#8466](https://github.com/apache/tvm/pull/8466) [#8571](https://github.com/apache/tvm/pull/8571)
- ARM: CMSIS-NN, Ethos-N [#8653](https://github.com/apache/tvm/pull/8653) [#7628](https://github.com/apache/tvm/pull/7628) [#8951](https://github.com/apache/tvm/pull/8951) [#7506](https://github.com/apache/tvm/pull/7506) [#7443](https://github.com/apache/tvm/pull/7443) [#7858](https://github.com/apache/tvm/pull/7858) [#6982](https://github.com/apache/tvm/pull/6982) [#8795](https://github.com/apache/tvm/pull/8795) [#8806](https://github.com/apache/tvm/pull/8806) [#8833](https://github.com/apache/tvm/pull/8833) [#9147](https://github.com/apache/tvm/pull/9147) [#9159](https://github.com/apache/tvm/pull/9159) [#9160](https://github.com/apache/tvm/pull/9160) [#9162](https://github.com/apache/tvm/pull/9162) [#9163](https://github.com/apache/tvm/pull/9163) [#9167](https://github.com/apache/tvm/pull/9167) [#9209](https://github.com/apache/tvm/pull/9209) [#9386](https://github.com/apache/tvm/pull/9386) [#9387](https://github.com/apache/tvm/pull/9387)
- Hexagon: build, compilation, model launcher, more target options and better runtime [#7784](https://github.com/apache/tvm/pull/7784) [#6718](https://github.com/apache/tvm/pull/6718) [#8821](https://github.com/apache/tvm/pull/8821) [#8822](https://github.com/apache/tvm/pull/8822) [#9033](https://github.com/apache/tvm/pull/9033) [#8823](https://github.com/apache/tvm/pull/8823) [#8859](https://github.com/apache/tvm/pull/8859) [#8865](https://github.com/apache/tvm/pull/8865) [#8915](https://github.com/apache/tvm/pull/8915) [#8954](https://github.com/apache/tvm/pull/8954) [#9024](https://github.com/apache/tvm/pull/9024) [#9025](https://github.com/apache/tvm/pull/9025) [#8960](https://github.com/apache/tvm/pull/8960) [#8986](https://github.com/apache/tvm/pull/8986) [#9010](https://github.com/apache/tvm/pull/9010) [#9011](https://github.com/apache/tvm/pull/9011) [#9189](https://github.com/apache/tvm/pull/9189) [#9220](https://github.com/apache/tvm/pull/9220) [#9355](https://github.com/apache/tvm/pull/9355) [#9356](https://github.com/apache/tvm/pull/9356)


- WASM: Update support for latest emcc, add ffi test. [#6751](https://github.com/apache/tvm/pull/6751)

#### BYOC Integration with Vendor Libraries: TensorRT, ACL, VitisAI

- TensorRT initial integration, stabilization, int8 calibration, dynamism support  [#6395](https://github.com/apache/tvm/pull/6395) [#7702](https://github.com/apache/tvm/pull/7702) [#7595](https://github.com/apache/tvm/pull/7595) [#7581](https://github.com/apache/tvm/pull/7581) [#7412](https://github.com/apache/tvm/pull/7412) [#7372](https://github.com/apache/tvm/pull/7372) [#9047](https://github.com/apache/tvm/pull/9047) [#8073](https://github.com/apache/tvm/pull/8073) [#8808](https://github.com/apache/tvm/pull/8808) [#6905](https://github.com/apache/tvm/pull/6905) [#7967](https://github.com/apache/tvm/pull/7967) [#8005](https://github.com/apache/tvm/pull/8005) [#8172](https://github.com/apache/tvm/pull/8172) [#8461](https://github.com/apache/tvm/pull/8461) [#8506](https://github.com/apache/tvm/pull/8506) [#8607](https://github.com/apache/tvm/pull/8607) [#7205](https://github.com/apache/tvm/pull/7205) [#7026](https://github.com/apache/tvm/pull/7026) [#7016](https://github.com/apache/tvm/pull/7016) [#7011](https://github.com/apache/tvm/pull/7011) [#6955](https://github.com/apache/tvm/pull/6955) [#6872](https://github.com/apache/tvm/pull/6872) [#7253](https://github.com/apache/tvm/pull/7253) [#6805](https://github.com/apache/tvm/pull/6805) [#9324](https://github.com/apache/tvm/pull/9324)
- Arm Compute Library (ACL) integration [#7649](https://github.com/apache/tvm/pull/7649) [#7206](https://github.com/apache/tvm/pull/7206) [#6532](https://github.com/apache/tvm/pull/6532) [#7121](https://github.com/apache/tvm/pull/7121) [#6724](https://github.com/apache/tvm/pull/6724) [#8149](https://github.com/apache/tvm/pull/8149) [#7251](https://github.com/apache/tvm/pull/7251) [#9396](https://github.com/apache/tvm/pull/9396)
- Verilator integration [#7406](https://github.com/apache/tvm/pull/7406) [#7351](https://github.com/apache/tvm/pull/7351) [#7286](https://github.com/apache/tvm/pull/7286) [#8094](https://github.com/apache/tvm/pull/8094)
- VitisAI integration [#6343](https://github.com/apache/tvm/pull/6343) [#7350](https://github.com/apache/tvm/pull/7350)
- BYOC infrastructure enhancement: improving control flow, AnnotateTarget, custom codegen [#6641](https://github.com/apache/tvm/pull/6641) [#6655](https://github.com/apache/tvm/pull/6655) [#6697](https://github.com/apache/tvm/pull/6697) [#6786](https://github.com/apache/tvm/pull/6786) [#7977](https://github.com/apache/tvm/pull/7977) [#8464](https://github.com/apache/tvm/pull/8464)


#### TVMC

- MacOS support [#8396](https://github.com/apache/tvm/pull/8396)
- AutoScheduler support [#7070](https://github.com/apache/tvm/pull/7070)
- Support cross compiler options [#7922](https://github.com/apache/tvm/pull/7922)
- Python scripting [#7823](https://github.com/apache/tvm/pull/7823) [#7698](https://github.com/apache/tvm/pull/7698)
- More flexible input specification [#7366](https://github.com/apache/tvm/pull/7366) [#7788](https://github.com/apache/tvm/pull/7788)
- More options, `--disable-pass` and `--config` [#7816](https://github.com/apache/tvm/pull/7816) [#8253](https://github.com/apache/tvm/pull/8253)
- Allow passing optional arguments to importers [#7674](https://github.com/apache/tvm/pull/7674)
- Model library format (MLF) support [#8086](https://github.com/apache/tvm/pull/8086) [#8331](https://github.com/apache/tvm/pull/8331)
- More backend and library support: metal, ACL, Vulkan, OpenCL, ROCm, Vitis AI [#8282](https://github.com/apache/tvm/pull/8282) [#7508](https://github.com/apache/tvm/pull/7508) [#8359](https://github.com/apache/tvm/pull/8359) [#6831](https://github.com/apache/tvm/pull/6831) [#8896](https://github.com/apache/tvm/pull/8896) [#7577](https://github.com/apache/tvm/pull/7577)
- Support for the new target system [#7651](https://github.com/apache/tvm/pull/7651) [#7654](https://github.com/apache/tvm/pull/7654) [#6788](https://github.com/apache/tvm/pull/6788) [#7304](https://github.com/apache/tvm/pull/7304) [#6855](https://github.com/apache/tvm/pull/6855)

#### Rust Binding

- Rust bindings installable via Cargo [#7503](https://github.com/apache/tvm/pull/7503) [#6678](https://github.com/apache/tvm/pull/6678) [#8631](https://github.com/apache/tvm/pull/8631) [#8665](https://github.com/apache/tvm/pull/8665)
- Initial support for diagnostic interface [#6656](https://github.com/apache/tvm/pull/6656)
- Fixes for using Python APIs from Rust [#7085](https://github.com/apache/tvm/pull/7085)
- Improve NDArray, GraphRt, Relay, IRModule, Array, Attrs bindings [#6563](https://github.com/apache/tvm/pull/6563) [#6741](https://github.com/apache/tvm/pull/6741) [#7138](https://github.com/apache/tvm/pull/7138) [#8353](https://github.com/apache/tvm/pull/8353) [#7082](https://github.com/apache/tvm/pull/7082)
- Improve error handling, error messages and fix memory leaks [#8289](https://github.com/apache/tvm/pull/8289) [#6815](https://github.com/apache/tvm/pull/6815) [#8714](https://github.com/apache/tvm/pull/8714) [#8725](https://github.com/apache/tvm/pull/8725)

#### Misc

- Enhanced CPP-RPC implementation: allow user supplied work dir, support of CPP-RPC server for Apple, support adb-shell style CPP-RPC [#7670](https://github.com/apache/tvm/pull/7670) [#8224](https://github.com/apache/tvm/pull/8224) [#8223](https://github.com/apache/tvm/pull/8223) [#7766](https://github.com/apache/tvm/pull/7766) [#7013](https://github.com/apache/tvm/pull/7013)
- Use PopenWorker to handle RPC system: [#7889](https://github.com/apache/tvm/pull/7889) [#7757](https://github.com/apache/tvm/pull/7757) [#7961](https://github.com/apache/tvm/pull/7961)
- Fold target host into target [#7462](https://github.com/apache/tvm/pull/7462) [#7791](https://github.com/apache/tvm/pull/7791) [#7534](https://github.com/apache/tvm/pull/7534) [#8835](https://github.com/apache/tvm/pull/8835)
- Target-based intrinsic lowering and legalization [#7936](https://github.com/apache/tvm/pull/7936) [#7809](https://github.com/apache/tvm/pull/7809)
- Add target tags for all existing CUDA GPU models [#7410](https://github.com/apache/tvm/pull/7410)
- Linear Congruential Random Engine [#8642](https://github.com/apache/tvm/pull/8642)

## 0.7
v0.7 brings many major features. The community works together to refactor the internal code base to bring an unified IR code structure with a unified IRModule, type system and pass infrastructure. We have also bought many exciting new features, some highlights include:

* Initial automatic scheduling support
* Initial command line driver interface
* WebGPU and webassembly support
* Better first class rust support in the codebase
* Intial Hexagon support
* Bring your own codegen (BYOC) support

The community also continues to bring high quality improvements to the existing modules including, but not limited to: better frontend coverage, performance, quantization, microTVM and dynamic shape support.

### New Features
#### Automatic Scheduling (Experimental)
* Phase 0: Ansor minimum system for auto schedule generating #5962
* Phase 1: Access Analyzer #6103
* Phase 1: Add `follow_split` and `follow_fused_split` steps #6142
* Phase 1: Add `pragma`/`storage_align`/`rfactor` steps #6141
* Phase 1: Add RPC Runner #6077
* Phase 1: Add `annotation`/`compute_at`/`compute_root`/`compute_inline` steps #6073
* Phase 1: Add `cache_read`/`cache_write` steps #6107
* Phase 1: Rename namspace form `auto_schedule` to `auto_scheduler` #6059
* Phase 1: The base class for cost models #6187
* Phase 1: feature extraction for cost models #6190
* Phase 1: XGBoost Cost Model #6270
* Phase 2: Basic GPU Sketch Search Policy #6269
* Phase 2: Evolutionary Search #6310
* Phase 2: Update heavy operations with `parallel_for` #6348
* Parallel the InitPopulation (#6512)
* Tutorial: Using the template-free auto-scheduler on CPU (#6488)

#### BYOC
* External codegen support in Relay (#4482)，(#4544)
* Bring Your Own Codegen Guide -- Part 1 #4602
* Bring Your Own Codegen Guide -- Part 2 #4718
* Relay annotation and partitioning for external compilers #4570
* JSON Runtime with DNNL End-to-End Flow #5919
* Handle one symbol for each runtime #5989
* Run accelerator specific optimizations #6068
* Arm Compute Library integration #5915
* Retire the example json runtime #6177
* `json_node.h` should include `data_type.h` #6224
* Improve installation tutorial #6170
* Add support for dense (fully connected) layer #6254
* Introduce the Ethos-N BYOC integration #6222
* Enable remote device via environment variables #6279
* Improved pooling support #6248
* Add support for quantized convolution #6335
* CoreML codegen #5634

#### Operator Coverage
* Add `strided_set` operation (#4303)
* Add support for conv3d (#4400), pool3d (#4478), 3d upsampling ops (#4584)
* Add group convolution for VTA (#4421)
* Add 1d deconvolution op (#4476)
* Allow batch matmul to be fused into injective ops (#4537)
* Add native depthtospace and spacetodepth operators (#4566)
* Add CUDNN conv3d support (#4418)
* Dilation2D operator support #5033
* Isfinite operator #4981
* Unravel Index operator #5082
* Add thrust support for nms #5116
* Resize3d, Upsample3d op support #5633
* Add operator Correlation #5628
* `affine_grid` and `grid_sample` #5657
* Sparse to dense operator #5447
* `Conv3d_transpose` op support added #5737
* add op `crop_and_resize` #4417
* Add bitwise ops #4815
* Sparse to dense operator #5447
* support dynamic NMS(Non Maximum Suppression), symbolic begin, end, and strides for strided_slice #4312
* `Conv3d_transpose` op support added #5737
* ReverseSequence operator #5495
* Conv1D #4639
* 1D Pooling #4663

#### Quantization
* Channel wise quantization - Quantize & Requantize #4629
* Support QNN ops. #5066
* Adding support for QNN subtract op #5153
* TFLite QNN Tutorial #5595
* Tutorial: Deploy Quantized Model on CUDA #4667
* Support asymmetric per-layer quantized operators #6109

#### Relay
* Add convertlayout pass in Relay (#4335, #4600)
* Added Merge Composite pass #4771
* Call graph for relay #4922
* Add inline pass #4927
* Target annotation for external codegen #4933
* GradientCell Relay Pass #5039
* Add MergeCompilerRegions pass #5134
* Non-recursive Graph Vistor and Rewriter (#4886)
* [Blocksparse] Pipeline for lowering dense model to sparse-dense (#5377)
* Relay op strategy #4644
* Static Tensor Array (#5103)
* Memory planner (part 1) #5144
* ONNX codegen #5052
* Add Parser 2.0 #5932, part 2 #6162
* Basic block normal form #6152
* Convert Layout pass. #4664
* Pattern Language, Matcher, Rewriter, and Function Paritioner #5231

#### Runtime and Backend
* Add ADTObject POD container type (#4346)
* TFLite RPC runtime (#4439)
* Standardized graph runtime export (#4532)
* MISRA-C compliant TVM runtime #3934
* Add String container #4628
* Introduce Virtual Memory Allocator to CRT (#5124)
* Initial implementation of Hexagon runtime support (#5252)
* FastRPC interface for Hexagon runtime (#5353)
* CoreML Runtime (#5283)
* AutoTVM + uTVM for Cortex-M7 (#5417)
* Windows Support for cpp_rpc (#4857)
* Implement TVMDSOOp(TensorFlow custom op) for TVM runtime (#4459)
* WebGPU support #5545
* TVM WebAssembly JS Runtime #5506
* Hexagon driver for offloading kernels to simulator #5492
* Introduce runtime::Array #5585
* Allow non-nullable ObjectRef, introduce Optional. (#5314)
* Introduce static slots for common objects. (#5423)
* ntroduce RValue reference(move) support to TypedPackedFunc (#5271)
* Introduce MetadataModule to separate code compilation/interpretation and weight initialization #5770
* Support module based interface runtime #5753
* Add TVM application extension with WASM runtime #5892
* Provide guide to user who has difficulty register SEqualReduce (#5300)

#### Rust Support
* Revive the Rust + SGX refactor #4976
* Improve Rust bindings: Map, Array, String, various IR nodes #6339
* Rust Refactor Stage 4: Rewrite Rust graph runtime to use new APIs #5830
* Second stage of Rust Refactor #5527
* tvm crate stage 3 of Rust refactor #5769
* Add first stage of updating and rewriting Rust bindings. #5526

#### TIR
* Introduce StructuralHash for the Unified IR. #5160
* Introduce StructuralEqual Infra for the unified IR. #5154
* Introduce ExprDeepEqual, Remove IRDeepCompare #5206
* [TIR] Introduce BufferLoad/Store (#5205)
* Improved massive build times caused by tir.floormod and tir.floordiv. Fixed Topi testcase. #5666
* Buffer logger assert removed #6147
* Enhance VerifyGPUCode #6194
* HoistIfThenElse added #6066
* Hybrid Script Support for TIR #6227
* Migrate Low-level Passes to Pass Manager #5198
* HoistIfThenElse added #6066
* Hybrid Script Support for TIR #6227
* Block scope hoisting added #6238

#### TE
* reverse-mode autodiff without any optimization #5121
* Tensor Expression Debug Display (TEDD) #4651
* Optimize and eliminate the Jacobian tensor for te.autodiff #6078

#### TVMC(Experimental)
* TVMC - A command line driver for TVM (Part 1) #6112
* TVMC - Linting error on onnx command line driver frontend #6536
* TVMC - Command line driver 'compile' (part 2/4) #6302
* TVMC - Introduce 'tune' subcommand (part 3/4) #6537
* TVMC - Introduce 'run' subcommand (part 4/4) #6578
* TVMC - Getting started tutorial for TVMC #6597


### Feature Improvement
#### Accelerator and Microcontroller Support
- Cleanup legacy verilog code (#4576)
- uTVM support for ARM STM32F746XX boards (#4274)
- Add --runtime=c, remove `micro_dev` target, enable LLVM backend #6145

#### Arithmetic Analysis
* Linear system and equation solver (#5171)
* Inequalities solver #5618
* Improve IntervalSet's floormod (#5367)
* Remove legacy const pattern functions (#5387)
* Handle likely in IRMutatorWithAnalyzer #5665
* ExtendedEuclidean merge impl to int_operator #5625
* Rewrite simplify fix for Vectorized Cooperative Fetching #5924

#### AutoTVM and Graph Tuner
* Adding ROCM schedules for TOPI (#4507)
* NHWC conv2d schedule templates for ARM (#3859)
* Use VM compile to extract autotvm tasks #4328
* Download fallback schedule file if it does not exist #4671
* Ignore error when removing tmpdir #4781
* Fix a bug in generating the search space #4779
* Minor bug fixes in AutoTVM for QNN graphs #4797
* Fix autotvm customized template #5034
* Add opt out operator for `has_multiple_inputs` for graph tuner #5000
* Customize SI prefix in logging (#5411)
* Update XGBoost verbosity option #5649
* Support range in index based tuners #4870
* Enable random fill and CPU cache flush for AutoTVM and Ansor (#6391)
* Auto-scheduler tutorial for GPU and necessary refactor/fix (#6512)

#### BYOC
* [BYOC] Bind constant tuples in graph partitioner (#5476)
* [BYOC] Add support for composite functions in BYOC (#5261)
* [BYOC] Register pattern tables from external codegens (#5262)
* [BYOC] Enhance partitioning and external codegen (#5310)
* [BYOC] Refine AnnotateTarget and MergeCompilerRegion Passes (#5277)
* [BYOC] Use Non-Recursive Visitor/Mutator (#5410)
* [BYOC] Refine DNNL Codegen (#5288)
* [BYOC] Add example of Composite + Annotate for DNNL fused op (#5272)
* [BYOC] Prevent duplicate outputs in subgraph Tuple (#5320)
* [BYOC] Introduce further operator support (#6355)
* [BYOC] Support input nodes with multiple entries (#6368)
* [BYOC] Add maximum support for float32 (#6506)

#### Codegen
* Intrinsic dispatching with OCML instead of LLVM for ROCm (#4499)
* Make target codegen take IRModule and PrimFunc. #5107
* Enhance CUDA codegen for SelectNode #4983
* Vectorization for intrinsics #5101
* [LLVM] Do not use `x86_vcvtph2ps_256` intrinsic with LLVM 11+ (#5267)
* [LLVM] Use llvm::ElementCount with LLVM 11+ when creating vectors (#5265)
* [LLVM] Use llvm::FunctionCallee in IRBuilder::CreateCall with LLVM 11+ (#5338)
* [LLVM] Include Support/Host.h for declaration of getDefaultTargetTriple (#5268)
* [LLVM] Replace calls to Type::getVectorNumElements (#5398)
* [LLVM] Use ArrayRef in calls to CreateShuffleVector (#5399)
* [LLVM] Use llvm::Align with LLVM 11+ to avoid warnings (#5264)
* [CodeGen] Cleanup generated code (#5424)
* Rename `target_id` => `target_kind` #6199
* 64-bit RPi4b target #6211
* Creating Target from JSON-like Configuration #6218
* Add python binding to new JSON target construction #6315
* Use target class in all codegens #6347
* Initial support for Hexagon codegen #6261
* Add --runtime=c, remove `micro_dev` target, enable LLVM backend #6145
* Add tvm::support::hexdump() debug utility #6154
* Adding AMD codegen unit tests (#4509)
* Support cuda tensorcore subbyte int data type in auto tensorcore #4546
* Handle empty LLVMModule in GetFunction #5146
* Support int4/int8 conv2d tensor core with HWNC layout #6121

#### Dynamism Support
* Add shape function for `zero`, `zeros_like`, `ones`, `ones_like` (#4448), `tile` (#4441)
* Support symbolic newshape for Reshape #5429
* Support symbolic TopK, Ones, Zeros and Full #5459
* Add `shape_of` instruction #5855
* symbolic `max_output_size` #5844
* Dynamic TopK Op #6008
* Dynamic `broadcast_to`, `zeros`, `ones` #6007
* Add dynamic reshape grad #6080
* Keep fixed dim when unifying dynamic shape #5795
* OneHot operation #6209
* Add Dynamic Resize Op #6198
* Dynamic full operator #6260
* Dynamic upsampling relay op #6273
* Dynamic Tile Op #5983

#### Frontend and User Interface
* TFLite parser support for `transpose_conv` (#4440), `unpack` (#4447)
* LLDB pretty printers for relay (#4453)
* ONNX to Relay converter op support: expand op (#4483)
* ONNX `auto_pad` in conv and convtranspose (#4563)
* TF to Relay converter op support (#4504) (#4551) (#4484)
* Remove unnecessary cast of constants in ONNX converter (#4573)
* Add support for tf.Keras networks in Relay Keras frontend #4630
* Add conv3d #4604
* Fix incorrect calculations in tf SLICE #4518
* Dynamically calculate `input_stats` of any `fake_quant` range #4789
* LSTM Support #4825
* Add `MIRROR_PAD` operator #4822
* use qnn helper function in softmax #4840
* Add Resize op converter #4838
* Add support for `TFLite_Detection_PostProcess` #4543
* Fix tests for tflite unary elemwise operations #4913
* GaussianDropout/Noise parsing support #4928
* Add parser support for 'square' operator #4915
* `make_loss` operator support #4930
* Add parser support for `l2_normalization` #4966
* ReadVariableOp operator support #4952
* Check graph inputs match expected #4992
* support multiply outputs #4980
* TFLite: Using real image for QNN testing. #4816
* TFLite: `FLOOR_MOD` & `FLOOR_DIV` support #4971
* PyTorch: Upsampling op support and enable registering a user defined op conversion map #4961
* PyTorch: fix unordered dictionary problem for python version under 3.6 #4982
* Operator support NonZero #5073
* Upsampling op support and enable registering a user defined op conversion map #4961
* Check graph inputs match expected #4992
* Add support for quantized models via QNN #4977
* Add initial control flow support #4964
* Remove FP32 piggy back and use QNN add/mul/concatenate #5061
* Add missing upcast to uint8 `avg_pool` conversion #5089
* Add initial 3D op support and test on Resnet 3D #5075
* Fix conv2d conversion for group conv (group > 1 but != in channels) #5132
* Add support for `max_pool1d` #5142
* Add support for split #5174
* `FLOOR_MOD` & `FLOOR_DIV` support #4971
* Activation functions support #4978
* Round op parsing support added #5022
* DepthToSpace and SpaceToDepth support #5041
* `TOP_K` op parser support #5051
* ReadVariableOp operator support #4952
* Support multiply outputs #4980
* `reduce_any` op parsing support #4926
* TensorFlow Parser Control Flow Enhancement #5020
* TensorFlow Frontend support with shared params #5042
* Support for AddV2 in Relay Tensorflow frontend converter. #5046
* conv3d frontend operator support #5080
* `max_pool3d` and Averagepool3d operator support #5085
* Support for Atan/Atan2 in Relay Tensorflow frontend converter. #5104
* Use leaky by default for LeakyReLU #5192
* Conv3D ONNX support and `conv3D_ncdhw` x86 schedules #4949
* Add support for FusedBatchNormV3 #5065
* Activations for pytorch #5194
* Dropouts And InstanceNorm support added #5203
* [Frontend] Asymmetric padding of convolution support (#4803)
* [ONNX]Pool3d & upsample3d op support (#5135)
* Add TopK to ONNX Frontend (#5441)
* Add RoiAlign to Onnx frontend (#5454)
* [PYTORCH]AvgPool3d, MaxPool3d and Squeeze op support (#5220)
* [PYTORCH]celu, gelu, selu activations (#5263)
* [Pytorch]layernorm bug fix and testcase updated (#5257)
* [PYTORCH]LayerNorm support added (#5249)
* [PYTORCH]GroupNorm op support added (#5358)
* [PYTORCH]Logical & Bitwise operator support (#5341)
* [PYTORCH]Tensor creation ops support (#5347)
* [PYTORCH]cosh,sinh,log2,log10,log1p op support (#5395)
* [PYTORCH]Rsub, Embedded, OneHot ops support (#5434)
* [PYTORCH]Abs, Arange, Softplus ops (#5295)
* [PYTORCH]isNan, isinf, isfinite, ceil, clamp, round ops (#5316)
* [PYTORCH]Activations for pytorch (#5194)
* [PYTORCH]Repeat, Reciprocal & Reshape Op support (#5280)
* [PYTORCH]`Reduce_ops` support added (#5308)
* [PYTORCH]Take, Topk op support (#5332)
* [PYTORCH]Dropouts And InstanceNorm support added (#5203)
* [PYTORCH]Unary Ops frontend support. (#5378)
* [Torch] Support Python list, more realistic recurrent networks (#5306)
* [PYTORCH]where, addcdiv, addcmul op support (#5383)
* [Torch] Add support for split (#5174)
* [Torch] Fix up graph input handling (#5204)
* [TFLITE]Logical not op support (#5475)
* [TFLITE]Hard Swish & MobilnetV3 model testing (#5239)
* [TFLITE]Gather, StridedSlice op support added (#4788)
* [TFLITE] Match TFLite shape for SSD custom op (#5473)
* Factor out import of common tflite.Operator in tflite frontend. (#5355)
* [TFLite] support for FILL and `SPLIT_V` operators (#5330)
* [TFLite] `L2_POOL_2D` operator (#5452)
* [TFLite] Add config option to specify FlatBuffers location (#5425)
* [TFLITE]Logical not op support (#5475)
* [TENSORFLOW]reduce ops updated (#5180)
* [TENSORFLOW] Fix `gather_nd` indices (#5279)
* [TensorFlow]Improve TensorFlow Static Shape Tensor Array (#5243)
* [KERAS]Minimum & AlphaDropout op support (#5380)
* [KERAS]Embedding layer (#5444)
* [KERAS]`Max_pool3d` and Averagepool3d operator support (#5085)
* [CAFFE2]add Mul and ConvTranspose operator (#5302)
* [MXNET]DepthToSpace & SpaceToDepth Operator (#5408)
* [MXNET]broadcast and logical op support (#5461)
* [MXNET] Use leaky by default for LeakyReLU (#5192)
* [MXNET] support elemwise logic ops (#5361)
* [Frontend|MXNet] SwapAxis operator support (#5246)
* [RELAY] Move frontend utils (#5345)
* [Pytorch] Fix translation of transpose when axis argument is as a list (#5451)
* LpPool Support added #5696
* Skip ADD inside Gemm op when vector is zero #5697
* ReduceL1, ReduceL2, ReduceSumSquare, ReduceLogSum ops added #5721
* MaxRoiPool, Mod & Xor op support added #5729
* Skip multiply with 1.0f constant for GEMM import #5800
* StatefulPartitionedCall/PartitionedCall Ops support added #5617
* Don't add cast for batch norm when type isn't changing #5731
* Conv3d Transpose OP added #5775
* expand bug fix #5576
* Support `max_pool2d_with_indices` #5549
* Add prim::device op #5584
* ImplicitTensorToNum support added #5603
* Matmul fix for `batch_matmul` #5604
* ReflectionPad2d op #5624
* Padding op support #5638
* Minor bug fixes #5683
* `floor_divide` support for squeezenet #5702
* ReplicationPad support added #5708
* aten::norm support added #5776
* broadcast and logical op support #5461
* MaxPool3d and AvgPool3d Ops support added #5614
* Softmin, trunc op support added #5715
* conv3d and `conv3d_transpose` addedx #5814
* Model importer to be compatible with tflite 2.1.0 #5497
* Nit: Function names made consistent #5515
* Select op support for tflite frontend #5486
* `GATHER_ND` #5508
* Quantize & Dequantize op #5394
* Fully connected op conversion made in sync with TFLite #5510
* `ADD_N` operator #5474
* onnx, mxnet, pytorch mathops added #5561
* abs, round, reciprocal, sign, softsign, `hard_sigmoid` ops support #5587
* Gather nd bug fix for one dim support in tensorflow #5588
* Add parser support for shape and range #5329
* Darknet support batch size for yolo #5688
* Improve Control Flow and TensorArray #5699
* MXNet: Softmin, trunc op support added #5715
* MXNet: conv3d and `conv3d_transpose` addedx #5814
* MXNet: Add parser for `contrib.box_decode` #5967
* Onnx: ReduceL1, ReduceL2, ReduceSumSquare, ReduceLogSum ops added #5721
* Onnx: MaxRoiPool, Mod & Xor op support added #5729
* Onnx: Skip multiply with 1.0f constant for GEMM import #5800
* Onnx: Fix an issue with #5755 and add Batch norm unit tests. #5845
* TensorFlow: StatefulPartitionedCall/PartitionedCall Ops support added #5617
* TensorFlow: Don’t add cast for batch norm when type isn’t changing #5731
* TensorFlow: Conv3d Transpose OP added #5775
* Add parser support for shape and range #5329
* Darknet support batch size for yolo #5688
* Improve Control Flow and TensorArray #5699
* Improve TF Parser to keep output nodes for `saved_model` #5794
* Add parser support for `relu6`, `leaky_relu`, `relu_n1_to_1`, `log_softmax` #4805
* Fix TF Dynamic input shape #5825
* Support a few contrib ops in mxnet #5819
* Improve TF Parser to keep output nodes for `saved_model` #5794
* Add parser support for `relu6`, `leaky_relu`, `relu_n1_to_1`, `log_softmax` #4805
* Check all unsupported ops before raising an exception #5929
* Add Pytorch advanced indexing #6318
* Support `index_select` #6295
* Fix cast to long #6301
* Fix dtype handling for modules with integer parameters #6311
* pytorch frontend support conv1d #6203
* Add cast to double, fix flatten conversion #6357
* Fix aten::max and aten::min conversion #6372
* Match pytorch 1.6 googlenet pretrained model (#6201) #6212Add unbiased variance op and corresponding support in pytorch frontend #6232
* Implemented PADV2 Operator for TFLite and added support for constant values in PAD. #6167
* Implemented `ONE_HOT` Operator for TFLite. #6223
* Implemented `EXPAND_DIMS` Operator for TFLite. #6243
* Implemented `REVERSE_V2` Operator for TFLite. #6304
* Implemented `MATRIX_SET_DIAG` Operator for Relay/TOPI and TFLite Frontend. #6303
* RESHAPE with dynamic shape arg in TFLite frontend #6208
* Constant input attr added to fully connected operation in TFLite frontend #6228
* Gather operation with indices as tensor expr in TFLite frontend #6168
* Added support for tflite quantized maximum and minimum #6018
* Unary ops support added in frontend #6196
* Introduce caffe frontend for tvm #6206
* Keras softmax and prelu fix under NHWC #6278
* add support for MXNET numpy operators #6054
* Refine tensorflow frontend 1.x & 2.x compatibility #6240
* Reduceops support added to frontend #6252
* Update precision in the ONNX `strided_slice`, update precision of ToScalar #6272
* NHWC import support. #4899
* Refine tensorflow frontend 1.x & 2.x compatibility #6240
* Fix node indices attribute error for tensorflow 2.3 #6288
* Support NMSv4 #6085
* Support for PyTorch Non-Maximum Suppression #6314
* ReplicationPad support added #5708
* MXNet pre-quantized BERT #6039
* Keep parameter names from PyTorch #5887
* Refine LSTMBlockCell to support dynamic rnn #5963

#### Relay
* Add function attributes to IR hash (#4479)
* Relay passes lookup overhead optimization (#4594)
* Add `half_pixel` option to Resize op #4610
* Skip example json runtime test when config is not set #4614
* Test `tensor_array` in vm #4608
* Improve `memory_allocation` pass to support multiple i/o dynamic kernels #4595
* Add unit test for `tensor_array_split` #4619
* Add parses support for unary elemwise ops #4634
* Add parses support for SLICE #4502
* Added pool autopadding and simplified converters. #4672
* Fix meaning of `conv2d_transpose` `output_padding` parameter #4318
* Use packed func macro for external codegen #4710
* Fix `_parse_param` bug #4711
* Add constant input support for elemwise ops #4666
* Add parser support for squared difference #4652
* Add type check to dense #4724
* Invoke tvm::build from relay `compile_engine` and interpreter #4723
* Broadcast condition, x, and y for Where op #4774
* Add parser support for relational ops #4695
* Remove duplicated BindParamByName function in VM compiler #4793
* Use SimplifyInference for L2 Normalization. #4795
* Expose vm OptimizeModule to Python #4800
* Add parser support for logical operators #4642
* Conv2D padding representation #4787
* Add support for quantized LOGISTIC #4696
* Fix VM compiler for while loop with free vars #4889
* Fix bug in re-processing call node in MergeComposite pass #4879
* Expose FunctionGetAttr to Python #4905
* Add a PyTorch to Relay Parser #4497
* Support data types for CSourceModuleCodegen args and output #4934
* Clean up and refactor PyTorch frontend #4944
* Relay pass to use fast exp/tanh #4873
* BatchNorm support with run-time mean and variance calculation #4990
* Reduce plevel of conv2d winograd implementation on cuda #4987
* Add operation tan to TVM #4938
* Outline and inline lifted functions for external codegen #4996
* Remove primitive attribute from composite function #5014
* Refactor Relay Python to use new FFI #5077
* Fix relay node registration after refactor #5083
* `Codegen_c.h` should include relay.function #5093
* Move expr.Function to function.py #5087
* Propagate constant to subgraphs #5094
* Adjust strategy plevel to achieve expected performance by default #5118
* Added a AnnotatedRegion utility class #5030
* Support TupleGetItem in body of pattern #5106
* Partition graph codestyle fixes #5202
* Re-wrote the Graph Partitioner to support multiple outputs #5143
* Fixes to MergeCompilerRegions #5195
* Refactor build module to take IRModule #4988
* Separate analysis and transform passes #5035
* Relay Node::make to constructor #5128
* relay::StructuralHash to tvm::StructuralHash #5166
* Conditions updated to cover better user scenarios #5043
* Replace UseDefaultCompiler with GetAttr #5088
* Return empty CSourceModule when no `lowered_funcs` exists in Relay mod #4847
* Clean up for memory pass to enable heterogenous execution support. (#5324)
* Remove re-exports of tvm.transform (#5337)
* [Refactor] Add memoized expr translator for use by backend codegen (#5325)
* Legalize - Use Non-recursive Rewriter. (#5296)
* Add additional check before re-using the cached match #5552
* Remove kCompiler attr from external functions #5615
* Pattern Language MergeComposite #5656
* Support Tuple Output in C/DNNL Codegen #5701
* Infer types in MergeComposite #5766
* Convert PatternGrouper to do pre-order, non-recursive analysis #5653
* Remove constants from partitioned functions #5663
* Add a check for null function attributes #5674
* Add ConstantPattern #5689
* Conditionally Embedding Constants in Partitioned Functions #5693
* Simplify Pattern API Implementations #5703
* Add ShapePattern and DataTypePattern #5760
* Remove unnecessary print #5642
* Improve Shape Func handling for Tuple inputs #5467
* Relay updated with String #5578
* Fix the creation of tuple of tuples in PartitionGraph #5616
* Preserve type information in Merge Composite #5640
* Move `compiler_begin`/`end_op` to local static objects #5622
* Fix `dataflow_pattern`.rewrite() hang if Match in IR #5680
* Fix segfault in pretty print when ObjectRef is null #5681
* Move `fallback_device` to config #5690
* Replace `build_config` with PassContext #5698
* Clear compile engine after task extraction #5724
* Add `storage_order` ignore in pooling layer. #5781
* Tweak cublas/cudnn priority level #5820
* Skip Unknown Function Symbols #5888
* Allow every runtime module to handle constants #5885
* handle Tuple/TupleGetItem in first order gradient #5946
* Add resnet-3d & Update network definitions for NHWC layout #5945
* Use TargetNode::attrs for Target serialization #5993
* each option of target str should only contain one ‘=’ #5988
* Rename `target_id` => `target_kind` #6199
* 64-bit RPi4b target #6211
* Add resnet-3d & Update network definitions for NHWC layout #5945
* Small bug fix for Conv1D imports. #5995
* Move `invoke_tvm_op` and `shape_func` to vm dialect #5958
* GRU Layer Support #6020
* Add pass for getting calibration data from a relay module #5997
* Merge two consecutive reshape ops #6052
* Add operation `scatter_add` to relay, based on scatter implementation. #6030
* i64 indices #5235
* Port `eliminate_common_subexpr` to non-recursive form #6134
* Fix interpreter for dyanmic shape input of `ndarray_size` #6086
* Allow to config allocator type and refactor vm code structure #6105
* Handle `ndarray_size` in FoldConstant #6156
* when converting constant nodes with types of int64 or float64 #6159
* Add ReshapeTensor instruction in the VM to replace the reshape op #6089
* Support combine multiple dense op just into dense #6062
* Add unbiased variance op and corresponding support in pytorch frontend #6232
* Specify additional layouts in convert layout pass #5422
* Safe check added for Merge Composite Call Node #5562
* Non recursive partitioning #5493
* Support combine multiple dense op just into dense #6062
* Make the max number of fused ops configurable #6327
* Implementation of the dynamic pad operator #6284
* change device annotation from post DFS to recursive #6124
* Make check stricter: disallow inserting function with free vars into module #6313
* Make check stricter by using Feature. Fixed multiple bugs #6326
* Resize support for NCHW-convertible layouts #6293
* Make AutoDiff thread through global function #6336
* Create Interpreter for each constant subgraph #6195
* Add Dynamic reshape to a dynamic namespace and add DynamicToStatic Pass #5826
* Expose relay BindParamsByName to Python #4751
* Implement pass manager tracing API #4782
* Move Ops in relay.op.contrib #4942
* Conditions updated to cover better user scenarios #4951
* [External codegen] Add test cases for fused ops with manual annotation (#4741)
* Multiple output support, reshape, split ops added #6296

#### Operator Coverage
* Allow empty tensor for `reshape`, `tile` and `strided_slice` #4618
* Fix meaning of `conv2d_transpose` `output_padding` parameter"; #4708
* Remove cpp upsampling and resize op #4769
* upsample operator 'NCHWinic' format support. #4791
* Injective schedule improvement #4786
* Enable vectorization on fp16 type #4867
* Support for Int8 schedules - CUDA/x86 #5031
* New PR to re-add tan to TVM #5025
* Register topi schedule for Relay `fast_exp` and `fast_tanh` #5131
* Move Dilation2d from nn to image namespace #5110
* Use Thrust sort for argsort and topk #5097
* Conv2d and Dense ops support on Tensor Core #5099
* Setting workload correctly for Depthwise Spatial conv ARM. #5182
* Adding a few missing math intrin #5011
* Missing vectorize for depthwise conv2d. #5196
* [TOPI] Using x86 schedules for ARM conv2d (#5334)
* [TOPI-ARM] Do not alter layout if layout is NHWC (#5350)
* [TOPI] Setting workload correctly for Depthwise Spatial conv ARM. (#5182)
* [OP] Add `fast_erf` implementation (#5241)
* [Topi] Tensorcore support for Conv3D (#5284)
* [intrin] a few more math functions (#5468)
* [Intrinsic] Add log1p, ldexp, atan2, hypot, nextafter, copysign (#5312)
* [topi] Add operation relay.nn.dilate() which calls topi.nn.dilate() (#5331)
* [Topi x86] Missing vectorize for depthwise conv2d. (#5196)
* [TOPI x86] Adding `unroll_kw` config option for depthwise conv2d. (#5197)
* [Topi] Breakdown topi.cc into smaller files (#5253)
* ReduceLogSumExp Operator support #5453
* Math ops added #5502
* Enable blocking format in x86 conv2d and fold scale axis #5357
* Add operation gather to relay. #5716
* Add `storage_order` ignore in pooling layer. #5781
* Fix bifrost spatial packing conv2d auto tune #5684
* Fix reshape usage in ARM schedule #5732
* Block sparse dense on cuda #5746
* Improve CUDA softmax scheduling #5600
* block sparse dense on cuda #5746
* pass-by-value -> pass-by-const-reference #5783
* Using MKL blas for quantized dense #6115
* topi -> tvm/topi #6186
* Use auto-tuner to improve `conv2d_gemm` performance #6117
* Improve CUDA `conv2d_transpose_nchw` #4762
* Add CUDA conv2d for NHWC layout #4737
* `conv3d_ndhwc` schedule #4775
* Fast exponent #4790
* Add Scatter to Topi/Relay/ONNX via hybrid script #5619
* Split MKL from BLAS. #6182
* Change the meaning of `conv3d_transpose` `output_padding` to match `conv{1,2}d_transpose` #6065
* Gather op support added #6013

#### Runtime and Backend
* Cythonize NDArray.copyto (#4549)
* Unified Object System runtime refactor (#4578, #4581, #4603)
* VM profiler: sort VM stats by time (#4601)
* Update RPC runtime to allow remote module as arg (#4462)
* Refactorying system lib and dso lib into library module (#4481)
* Improve TSIM virtual memory mapping (#4545)
* make adt tag signed #4605
* Improve TVMBackendPackedCFunc to allow return val #4637
* EdgeTPU runtime for Coral Boards #4698
* Fix memory leak when using openMP #4811
* Fix memory leakage of TVMByteArray #4856
* Fix `TVM_DLL_EXPORT_TYPED_FUNC` to work on Windows #4955
* Fix memory leak when using openMP #4811
* Export GraphRuntime in `tvm_runtime.dll` #5002
* MISRA-C compliant TVM runtime #3934
* Update the `type_keys` to reflect the code-org #5074
* Fix AttrEqual for Array and StrMap, double #5054
* Export GraphRuntime in `tvm_runtime.dll` #5002
* Fix unused-value warning #5140
* crt error handling #5147
* Bundle deployment with static linking #5158
* Implemented kDLCPUPinned (cudaMallocHost) #4985
* Explicitly cast min/max operands #5090
* `ref_counter` -> `ref_counter_` #5184
* Expose runtime::String to Python (#5212)
* [FFI] Refactor runtime.String to subclass str (#5426)
* [RUNTIME] Auto conversion from str to runtime::String in PackedFUnc (#5251)
* [RUNTIME] Improved Packed FFI for optional. (#5478)
* [Hexagon] Add `hexagon_posix.cc` to TVM/RT sources in the right place (#5346)
* [FFI] Refactor runtime.String to subclass str (#5426)
* Fix workspace #5503
* Store nullptr PackedFunc as nullptr for better error propagation #5540
* Improve PackedFunc robustness #5517
* Seg fault in WorkspacePool's destructor (#5632) #5636
* Resolve constexpr issue in debug mode. #5651
* Add `compile_shared` option to linux compile utility fn #5751
* Call sync in CopyFromRemote and CopyToRemote #5512
* Fix the multihop cpu case #5522
* Improve RPCServer AsyncIO support. #5544
* Modularize the RPC infra #5484
* Add `compile_shared` option to linux compile utility fn #5751
* Overload string operators #5806
* Only initialize required module #5926
* if a param not in input, we should still consume it’s data #5990
* init TVMPackedFunc’s name #6044
* Enable auto conversion `String->DLDataType` #6214
* Support random fill #5913
* Use new to avoid exit-time de-allocation order #6292
* Add `parallel_for` support to run a loop in parallel #6275
* Solve ARM BIG.LITTLE heterogeneous multicores #4747
* [RUNTIME] Quick fix PackedFunc String passing (#5266)
* Introduce runtime::String::CanConvertFrom #5718
* Restore the StrMap behavior in JSON/SHash/SEqual #5719
* Support overriding RPCWatchdog termination behavior on Android and other platforms #6216
* Set `NDArray::Container.shape_` in NDArray::FromDLPack (#5301)
* Enable x86 cpu cache flush #5914

#### Quantization
* Conv2D type checking for kernel per-channel scales. #4732
* Add missing nullptr check #4773
* Doc fix on convolution and dequantize #4799
* Conv2D with dilation support. #4796
* Making `scale`/`zero_points` as expr instead of attrs. #4611
* Make calibration faster and more memory usage friendly #4589
* Doc fix on convolution and dequantize #4799
* Conv2D with dilation support. #4796
* Optimize lowering for requantize and FixedPointMultiply. #4798
* More doc fix on quantize and convolution #4874
* Add support for per channel weight scale in dense op #4880
* Add support for quantized models via QNN #4977 #5013
* Support 4D padding. #5036
* [Requantize] Cleanup and Optimize Lowering (#5286)
* [Topi, ARM] Disbale Winograd for quantized tensors. (#5363)
* Adding support for TFLite QnnSubtract operator. (#5230)
* Remove developer facing api from frontend exports. (#5375)
* Add Quantize/Dequantize Partitioning #5940
* Add support for quantized models via QNN #5016
* Quanitze operation expanded to take const argument #6127
* FP32 and Quantized Object Detection Model #5479
* Support CallNode inputs in qnn.concatenate #5360
* QNN support for TFLite 2.1.0 quantized models #5848

#### TE
* Tighten split's extent #4931
* Set split node's range to minimum of ext and split factor or split np… #5044
* Support mixing normal and cross-thread reduction (#5193)
* Inline -> `te/schedule/operation_inline.h` (#5386)
* Create loops according to storage scope and thread hierarchies (#5190)
* Fix import in dump pass ir (#5327)
* Scalar support for te.extern #6079

#### TIR
* IR readability enhancement (#4501)
* Introduce tir::PrimFunc #5070
* Introduce PrimFuncPass. #5139
* [TIR] Enhance Substitute, python bindings for Substitute/PostOrderVisit (#5400)
* [TIR] Remove ProducerConsumer and `AllocateNode::new_expr` (#5333)
* [TRANSFORM] Enable CopyOnWrite for TIR passes. (#5309)
* [REFACTOR] Migrate LowerTVMBuiltin, InferFragment, LowerThreadAllreduce, ThreadSync to Pass Manager (#5213)
* [REFACTOR] Remove te::Tensor dependencies from TIR passes. (#5372)
* [TIR] Refactor MakePackedAPI to target dependent stage. (#5326)
* [REFACTOR] tvm.hybrid -> te.hybrid (#5223)
* [REFACTOR] Migrate most of low-level build to use the Pass Manager. (#5225)
* [REFACTOR] Migrate low-level passes in tvm.lower to the Pass Manager (#5364)
* [TIR] Migrate VTA TIR passes to the new pass manager. (#5397)
* [REFACTOR] Migrate all low-level passes to the Pass Manager. (#5233)
* [REFACTOR] Introduce ExprDeepEqual, Remove IRDeepCompare (#5206)
* [REFACTOR] RewriteForTensorCore -> te/schedule (#5379)
* [REFACTOR] Remove `ir_pass` in favor of analysis/transform. (#5415)
* text format printer considering future parsing use #5483
* Remove buffer params from pass config. #5652
* std::string -> String Migration in TIR nodes #5596
* Remove `CallNode.call_type` in favor of attribute. #5937
* Remove legacy HoistIfThenElse #5944
* Improve Let/LetStmt support. #5949
* Refine side effect analysis. #5954
* `Provide->ProducerStore`, `Realize->ProducerRealize`. #5750
* Migrate the tvm/tir/expr.h to constructor #5773
* Migrate tir/stmt.h to use constructor. #5778
* Cleanup unused classes #5789
* Add tir prefix to type keys #5802
* Enhance VerifyGPUCode #6194
* Enforce buffer pointer var type to be consistent with dtype. #6317
* Create a StringImm reference type #4806
* Add init member to ReduceNode #6138
* Add dump and print for debugging (NFC) #5207
* Streamline Function Attr interface. #5045
* `alpha_equal` to `structural_equal` #5161
* Remove AttrsEqual and AttrsHash related code #5169
* [NODE] General serialzation of leaf objects into bytes. (#5299)
* [POC] Initial stab at `std::string->String` upgrade (#5438)
* [TIR] Make `lower_warp_memory` support `extent(threadIdx.x) < warp_size` (#5307)
* [PASS] dtype rewrite for indexing variables (#5092)
* [PYTHON] Enhance `with_attr` API, cleanup MakeAPILegacy in testcases (#5335)
* [PYTHON] Make IntImm more like an integer (#5232)
* [IR] Move to runtime::String (#5276)
* [IR] kExternalSymbol -> kGlobalSymbol (#5211)
* [IR] Remove PrimExpr from String (#5311)
* IRModule is updated with String #5523
* IR is updated with String #5547
* Streamline ir/op Registry #5609
* Migrate IRModule ObjectRef to not-null #5654
* Migrate BuildConfig to PassContext. #5668
* relay.op.Op -> tvm.ir.Op #5705
* Separate ArgTypeCode from DLDataTypeCode #5730
* Remove legacy `compute_expr.h` #5738
* Call::Halide => ProducerLoad, DSL/TIR decouple. #5743
* `Provide->ProducerStore`, `Realize->ProducerRealize`. #5750
* Migrate the tvm/tir/expr.h to constructor #5773
* Migrate tir/stmt.h to use constructor. #5778
* Migrate all Object construction to constructor. #5784
* Cleanup unused classes #5789
* Finish `std::string->String` updates #5793
* Add tir prefix to type keys #5802
* Change Call.name to Call.op(RelayExpr) #5863
* Range/IntSet API style consistency. #5953
* Separate ArgTypeCode from DLDataTypeCode #5730
* Migrate all Object construction to constructor. #5784
* Finish `std::string->String` updates #5793
* Unify StrMapNode and MapNode #5687

#### Performance Improvements
* Int8 GEMM performance enhancement using Cublas (#4550)
* Speedup TSIM with multi-threading (#4491)
* Support cudnn softmax (#5214)
* Add cuDNN grouped convolution support (#5319)
* Winograd support for Conv3D (#5186)
* Improve `get_valid_count` and nms performance for CUDA (#5339)
* Optimizations of `global_ave_pool` for NHWC layout (#5450)
* Optimization of Conv2d Winograd algorithm on Tensor #5485
* Some performance improvement to VM #5901
* Optimize x86 `conv3d_ndhwc` using data packing approach. #4866
* Improve NHWC depthwise convolution for AArch64 #6095
* Improve quantized convolution performance for armv8 architectures #5754

#### Documentation
* Adding benchmark log format doc (#4366)
* Add Ninja build system to installation docs (#4554)
* Doc/comment fixes (#4452, #4463, #4469, #4493, #4397, #4580, #4585, #4591)
* Fix doc after moving to unified IR #4835
* Introduction to module serialization #4564
* ConvertLayout - Call RemoveUnunsedFunctions. #4834
* Fix bugs that override `n_trials` #4842
* Update the vm doc #4868
* Refine the example description of `max/min/sum/tag_scope` #4974
* Fix vta tutorial #4809
* Introduce how to add hardware backend to FAQ #4898
* Update API docs to reflect the status after the refactor. #4907
* Fix sphinx warnings #4917
* Fix Sphinx Warnings (RST indent, cross-ref, and image scale) #4920
* Fix Sphinx Warning: the target found for cross-reference #4925
* Sphinx -- Introduce alias detection. #4954
* Fix Warnings from #4942 #4959
* Fix sphinx precheck #4967
* Move `git_howto` to rst, add Stage documents to te #5055
* Add doc for Relay op strategy #5078
* Update relay docs #5112
* Include a tarball of docs, add a security faq #5119
* Cleanup docs before rebuild #5127
* Minimize necessary doc change #5129
* Various sphinx related fix. #5168
* Point docs to the ASF site. #5178
* Use https link #5183
* Reduce artifcats generated by sphinx gallery #5208
* Refine the example description of `max/min/sum/tag_scope` #4974
* Description updated for pooling attributes #5091
* [DOCS] Migrate some markdowns to rst, fix sphinx3 warnings (#5416)
* [DOCS] Misc docs improvements (#5222)
* [DOCS] Bring relay docs to the top-level flat view (#5343)
* [DOCS] Reduce artifcats generated by sphinx gallery (#5208)
* [DOCS] Use https link (#5183)
* [DOCSTRING]missing function parameters updated (#5228)
* [DOCS] Migrate HLS documents from md to rst (#5419)
* [Tutorial, QNN] Add tutorial for loading quantized PyTorch model (#5321)
* [Docs] VTA install doc migration from md to rst (#5442)
* [Docs] compiler version in docs (#5281)
* Remove legacy `compute_expr.h` #5738
* `TVM_REGISTER_API` -> `TVM_REGISTER_GLOBAL` #4768

#### Bug Fixes
* Add bfloat16 typeflag support (#4525)
* MSVC / Windows fixes (#4455, #4569)
* Fix Makefile for `howto_deploy` (#4457)
* Fix GCC 4.8 compact (#4461)
* Fix search path to build `libtvm_topi.so` (#4467)
* Fix for `conv2d_transpose` CUDA compilation (#4472)
* Fix for LLVM 10.0 codegen (#4480, #4515)
* Fix alter op layout when calling global var (#4454)
* Fix `float2half_rn` support for cuda compute capabilities < 53 (#4489)
* Fix compile errors for OpenCL backends (#4492)
* Fix serialization precision loss (#4503)
* Fix hybrid script to support array of tensors (#4494)
* Fix annotation for multiply op (#4458)
* Fix Dockerfile for linter CI (#4506)
* Fix TF resize for dynamic size models (#4510)
* Fix `bias_add` gradient (#4516)
* Fix tanH unit test function call (#4517)
* Fix extra reshape parameter for ONNX (#4524)
* Fix crash caused by empty TOPI config (#4520)
* Fix ONNX shape op type to use int64 (#4528)
* Fix crash in TSIM virtual memory driver (#4527)
* Replace deprecated python library in setup script (#4533)
* Fix NMS `max_output_size` loop (#4541)
* Fix style in IR mutator and IR visitor (#4561)
* Fix compiler warning (#4559)
* Fix to get end to end inference on Chisel VTA (#4574)
* Fix LLVM build by adding missing intrinsics headers (#4575)
* Fix context creation in quantization (#4582)
* Fix NDArray SaveDLTensor signature (#4586)
* Fix dense pack schedule for x86 (#4539)
* Fix for broadcast tensor of scalar type (#4577)
* Datatype refactor (#4513, #4560)
* Add const qualifiers for NDArray container (#4590)
* Fix TF <= 1.12 compatibility (#4593)
* Fix for graph debug runtime (#4598)
* Disable copy constructor for external codegen (#4597)
* Make ADT tag signed (#4605)
* Added declare of aluBits for TensorAlu #4624
* Get around limitation of g++-4.8 #4626
* Bugfix StmtMutator IfThenElse #4609
* Remove unecessary rdynamic #4613
* Resolve constexpr related link error in debug mode #4641
* Asymmetric padding #4511
* Reduce data size of asymmetric padding testcase #4658
* Fix Base64OutStream portability issue #4668
* Fix `topi.nn.global_pool` layout="NHWC" #4656
* Also package core.rly #4679
* fskip of EliminateCommonSubexpr cannot always return false #4620
* Fix Python syntax error in `start_rpc_server_to_tracker.py` #4682
* os.path --> osp to match the import #4681
* GitHub actions/checkout@v1 --> v2 #4680
* Fix Python syntax error AGAIN in `start_rpc_server_to_tracker.py` #4685
* Use ==/!= to compare str, bytes, and int literals #4686
* Rename `start_rpc_server_to_tracker.py` to `start_rpc_server_to_tracker.sh` #4689
* GitHub Action lint Python code for syntax errors #4688
* Generate blob use LLVM directly #4657
* Reduce input size to fix oom #4653
* Fix RemoveUnusedFunctions pass #4700
* Link the math library by default #4713
* Update mainline version to 0.7.dev0 #4720
* Add SizeVar representing non-neg valued variable in a tensor shape #4684
* Fix the compile problem of `cpp_rpc` #4725
* JSON upgrader to upgrade serialized json. #4730
* Fallback schedule for Int8 depthwise. #4733
* Fix dense x86 schedule #4728
* Fix demo dockerfile build failed #4744
* Improve CUDA vectorizer #4736
* Add .asf.yaml for github info #4761
* Fix padding in pooling op #4738
* Remove `run_infer_type` duplicates #4766
* pooling.cc improvements #4767
* Export `builtin_fp16` on Windows #4731
* Fix Tensorflow conv3d pad bug, add non-cubic data and kernel tests #4772
* Bump prebuilt-image version in demo dockerfile #4770
* Update `tune_simple_template.py` #4778
* Explicitly link to cublasLt if it exists #4776
* Fix hasattr by extracting Python error type from Windows error message #4780
* Replace os.path.exists with try...except...else #4784
* Make sure to visit the arguments of inlined functions #4783
* Parse additional exception strings #4785
* Fix #4670: add bias for fc layer #4801
* Change color channel from BGR to RGB for darknet preprocessing #4794
* Fix -Wextra #4804
* Fix vta tutorial #4809
* Minor bug fixes in AutoTVM for QNN graphs #4797
* Fixed subprocess creation under windows #4820
* Improve tol to resolve flaky case #4836
* Fixed process termination routine in windows #4844
* `test_cuddn` flaky #4846
* Mxnet parser for Qnn dialect #4714
* Enhance `cc.cross_compiler` #4817
* Fixed crash caused by reversing bitwise operations #4852
* Reverse some changes made for `intel_graphics/conv2d.py` in PR #4849 #4853
* const auto p -> const auto& p #4861
* Fix onnx import bugs #4750
* Explicit llvm::StringRef to std::string conversion #4859
* Update the runtime PackedFunc for module #4871
* Improve antlr import error message #4888
* Fix `alpha_equal` bug for attribute check #4897
* Fix issues in cuda codegen #4876
* Fixed: Bitwise ops on floats causing wrong code generation and crashes. #4892
* Fix `tvm.target.generic_func` runtime detection #4910
* `topi/tests/python/test_topi_sort.py::test_argsort` #4891
* Use opencv reisze method for preprocessing of image in darknet #4883
* Fix build breaks with StringRef changes #4923
* Remove unnecessary spliting in the cached chunk #4935
* Fixing an Infinite Loop case in UnmatchedChecker. #4881
* Remove SGX toolchain installation from CI Dockerfile #4948
* Fix tedd tutorial after strategy change #4947
* Allow customize MKLDNN library location #4814
* Added CopyFromBytes and CopyToBytes convenience methods to NDArray. Fixed typos. #4970
* Fix gcn tutorial failure #4994
* Fix stride default value None in torch.nn.functional.avg_pool #4984
* Fix ROCm strategy for winograd conv selection #5001
* Fix `get_valid_count` flaky test for cuda #4901
* Change Scala Linter scalafmt => scalastyle #4998
* Kill from tvm import te #5007
* Chisel fixes and de10nano support #4986
* Fix gpu not found when running TVM docker #4975
* Fixes for pylint==2.4.4 #4849
* Fix unordered dictionary problem for python version under 3.6 #4982
* Fix gcn tutorial failure #4994
* Fix stride default value None in `torch.nn.functional.avg_pool` #4984
* Fix ROCm strategy for winograd conv selection #5001
* Early checking added and new test cases added for schedule fuse #5010
* Fixed div by zero core dump. Fixed rounding intrinsics on int crash #5026
* Test case modified for int type #5012
* Bug Fix for ARM CPUs. Lower strict assumption. #5063
* Triage the testcases to fit the the new namespaces #5071
* Add colors to `compute_at` edges and thread/block indices. #5111
* Temporary fix to the stack overflow issue in autotvm task extraction #5019
* Fix compilation of If-Elses #5040
* Fix CompilerAttrs #5109
* Fix the existing test cases before refactoring. #5122
* Fixed bug where shifting by out-of-bounds value results in no compute code being emitted. #5115
* Fix for issue #4831. The `data_min_idx` and `data_max_idx` were flipped. #5136
* Duplicate likely nodes added when loop axis split unevenly #5084
* Fix incorrect name of calibration mode #5150
* Remove contrib spatial pack schedule of depthwise convolution #5148
* Fix annotate pass static variable #5023
* Fixed ConvTranspose2D parsing #5157
* Nullptr check #5176
* rocm: fix miopen convolutions #5179
* rocm: fix `dense_rocblas` in strategy, topi #5191
* Fix CRT static test bug (#5293)
* Fix perf regression of tir refactor (#5258)
* Bugfix in tensorflow `space_to_batch_nd` (#5175)
* Compilation warnings fixed for 32bit and 64bit compilation (#5349)
* Fix hang in MergeCompilerRegions (#5227)
* Fixes to MergeCompilerRegions (#5195)
* Fix generation of LLVM intrinsics (#5282)
* Fix setting up hints for getaddrinfo (#2872)
* Add ConstantNode to IsAtomic (#5457)
* Fix String SEqual (#5275)
* Fix fuse over functions that are handled by external codegen (#5365)
* Fix memory leak when accessing NDArray (#5413)
* Remove the duplicate PrintIR pass in Relay (#5403)
* Fix `lower_warp_memory` (#5247)
* Fix `lower_warp_memory` when there are >1 warp buffers (#5368)
* Fix intel conv2d auto tune (#5200)
* Fix FuseBatchNorm output cast error if `need_cast` is True #4894
* Fix an assertion exposed by loop vectorizer #4916
* Fix error message #4945
* Fix for recursive let #5757
* Fix Calibration Pass to Support Modules with Multiple Functions #5768
* Fix what looks like bizzare copy-paste issue #6010
* Fix bug in `transpose_shape_func` #6180
* Fix bugs in CUDA codegen (#5209)
* Don’t remove() TemporaryFile in del. (#5414)
* Fix `test_ir_type`. (#5390)
* Fix multiple identical inputs bug (#5389)
* Add cuda target check to dense tensorcore schedule. (#5376)
* T2 test fixups (#5391)
* Fix miopen padding (#5433)
* Misc fixes for ROCm (#5431)
* Fix copy constructor (#5237)
* Corrected TVM autotuning on GPU (#5432)
* Fix vector load (#5226)
* Minor bugfix in `message_passing.cc` (#5254)
* Fix a bug when vectorized load&store was involved for… (#5428)
* Fix to skip node not in graph. (#5238)
* Fix #5388 [VULKAN] vkBuffer released before memory copy command se… (#5418)
* Fix a minor error in `device_annotation` (#5291)
* Fix scalar’s ndim is 0 (#5344)
* Fix the runtime raise error #5586
* Fixed bug in attribute parsing for pool layers. #5582
* AutoTVM incorrect measurement #5511
* fix a min/max simplify bug #5761
* Rename `tvm_dso_op` to `libtvm_dso_op` #5714
* Fix generating types like float44 and float88 #5722
* Avoid downloading when `TOPHUB_LOCATION` is NONE #5720
* codegen llvm: move nvptx-specific intrinsic handling into `codegen_nvptx` #5726
* ROCm warp shuffles and reductions #5727
* fix small bug about `dense_grad` #5695
* Clarify downstream consistency of TVMArgTypeCode #5742
* Fix gelu in PyTorch frontend, tighten numerical checks #5763
* Make batch matrix multiplication on GPU tunable #5752
* update vulkan build rule #5777
* aten::norm support added #5776
* Edit onnx parser to infer values in post order #5755
* Support symbolic inputs of Fill #5762
* support `aten::type_as` in the pytorch frontend #5787
* Temporary disable fp16 `type_as` test for PyTorch Frontend #5799
* Add config switch for nn.dense layer type. #5801
* Move cpu-only frontend tests to a CPU stage #5807
* Pin hand landmark network to version 0.7.4. #5813
* Limit number of threads in all jobs #5815
* Error msg update #5818
* fix relay.build to not change the module argument in place #5822
* Fix InferType when module contains Prelude #5797
* Add a combine `batch_matmul` pass #5791
* RepeatVector, Conv3DTranspose op support added #5833
* Fix converting serialized quantized models #5839
* ffi (Object): make class dict visible in instances #5843
* Additional canonicalization added for AddNode #5846
* Suppress the warning messages when compile engine selects impls #5821
* fix #5849 #5851
* Introduce POD-C Compliant tvm::Map #5740
* Add bfloat16 #5601
* Add Python Classes for all Attrs #5853
* Fix map assign issue in CI test #5854
* Introduce Target Id Registry #5838
* Update `has_dtype/has_shape` to pattern lang doc #5847
* Add `nn.batch_flatten` as quantizable. #5805
* Fail early before running invalid dynamic graphs #5856
* Improve type handling in PyTorch frontend #5834
* HotFix the python intrin rule #5895
* add a few gradients #5899
* Add Binary Intrinsic ops to TIR Ops in C++ #5900
* Allow implicit conversion in TVM FFI to tvm::Bool #5907
* PyTorch frontend: fix handling of duplicate use of a model weight #5897
* Don’t multiply by constant 1 uselessly in dense #5911
* Support any index matching for TupleGetItem #5909
* Add MicroTVM tutorial using the STM32F746 discovery board #5655
* Fix serialization of inf float value #5912
* Fix CPU Thread Binding for Multiple Sockets #5918
* CUDA device API & VerifyGPUCode pass update #5898
* Update install.rst #5858
* Two small fixes to AMDCPU codegen for LLVM 10+ and ROCm 3.5+ #5920
* Add LegalizeInvalidAttach to legalize the `compute_at` location after split or fuse #591
* Don’t rewrite expressions used outside of the pattern #5930
* Add TupleGetItem to CSE #5931
* Various update for CoreML codegen #5934
* Update date in the NOTICE #5943
* Raise right error in tensorflow split op #5951
* Add rm xla attributes in tf docs #5950
* Fix OpenCL `get_valid_counts` errors due to intrinsic `atomic_add` #5857
* Amendments for gradients #5941
* Fix the meaning of `conv{1,2}d_transpose` `output_padding` parameter. #5758
* Make first order gradient graphs more efficient #5959
* Raise an exception when extern function does not return Stmt #5964
* Improve docker/bash.sh to handle git worktrees #5970
* Install DNNL (OneDNN) to CI Environment #5936
* Add Dynamic reshape to a dynamic namespace and add DynamicToStatic Pass #5826
* Add meshgrid op in Relay, TOPI, Pytorch frontend #5961
* Print right number of parentheses for LoadNode #5965
* Migrate data structure of TargetNode #5960
* Remove redundant function CreateBufferVecPtr #5982
* Fix string argument mismatch in GraphRuntimeCodegen #5933
* VectorType::get with two parameters is deprecated in LLVM 11+ #5984
* Fix Compilation Error in CRT #5713
* Fix runtime::String backward compatibility in JSON #5725
* Allow RPCWrappedFunc to rewrite runtime::String as std::string #5796
* Fix reshape #5739
* Fix building with LLVM-10 on macOS #5859
* Add cuda 11 to `contrib.nvcc.find_libdevice_path()` #5902
* Fix sequential cpp test #5745
* Infer types in MergeComposite #5766
* Fix recursive let for well formed check #5780
* Recover global state after `test_util.py` #5824
* Fix bug in rpc ring buffer shrink #5516
* Fix remote device sync #5538
* Fix bug in rpc ring buffer shrink (#5516) #5537
* RPC Server error fix on Pynq FPGA #5607
* Fix FloorMod Simplifier #5509
* Fix Python debugger segfaults with TVM built with LLVM #5685
* Fix Compilation Error in CRT #5713
* Fix runtime::String backward compatibility in JSON #5725
* Allow RPCWrappedFunc to rewrite runtime::String as std::string #5796
* Fix reshape #5739
* Make "none" DataType explicit #5491
* Change "scalar" and "stack" in IDL from "inrout" to "in" #5487
* Link necessary libraries when building runtime for Android #5496
* Fixes for wasm32 target #5489
* Reset target and wait for runtime initialization on connect. #5499
* Bump tophub rocm version #5504
* Improve commentary for RingBuffer #5518
* Add unit tests for ONNX PRelu and fix importer to pass them. #5521
* LRN only supports 4D tensors, remove it from `alter_op_layout` #5520
* Fix an issue with ONNX Upsample #5530
* Cache PrimExpr instead of raw pointers in bound analyzer #5533
* fix a few bugs with shape inference and types in the ONNX importer #5534
* Add Onnx Pad v11 #5539
* Changes to `cpp_rpc` to make it work on Android (+ Hexagon offloading) #5535
* Fix to reduce RAM size during loading model #5507
* Fix MakeLoopNest for warp memory #5382
* Load platform specific lib for tvmdsoop instead of the hard-coded tvm_dso_op.so #5542
* Add tests for running micro on native arm hardware #5546
* Apparently, ONNX Conv with no 'pads' defaults to zero padding #5548
* clang-format the h,cc,m files. #5557
* Fix conv2d alter op for arm cpu #5532
* Fix topi test for non tensorcore CI. #5563
* Add clang-format and nodejs to ci-lint #5567
* Enable clang-format. #5572
* Allow `ubuntu_install_darknet.sh` to work in both 18.04 and 16.04 #5574
* Add a quantized conv2 unit test for the tflite front-end #5558
* Fix JSON graph dumping. #5591
* Warp level reduction support for CUDA #5498
* One more fix for concurrency count #5589
* Improve robustness of the docs build #5583
* Phase out WebGL #5570
* Fix vulkansdk in the ci-gpu and upgrade to 1.2.135 #5566
* Update ci-cpu to bionic #5554
* Overestimate binary size for microTVM compiled binaries. #5590
* Fix bug and re-enable RPC execution test #5436
* Add ostream formatters for TargetPtr/TargetVal. #5592
* Fix cross thread reduction #5551
* Fix TVMArray layout on device #5599
* Add debug mode to tempdir() #5581
* Represent alignment information in LLVM IR #5598
* Fix codegen for warp shuffle intrinsics #5606
* Fix Topological Order calculation for DFPattern Language #5612
* Global MaxPool3d and AvgPool3d support #5098
* Fix build error of iOS RPC #5621
* isn't a CallNode sometimes #5623
* Introduce config to PassContext. #5631
* CMAKE fix #5630
* Label Pattern Partitions #5627
* Extend AttrPattern to support CallNode and FunctionNode attributes #5637
* Increase bss section size. #5660
* Add buffer name when creating tensor bindings #5670
* µtvm debug improvements #5648
* enable `amd_apu` device on vulkan target #5659
* Support TupleWrapper as direct ancestor of control flow ops #5639
* add tvm.micro pydoc to sphinx #5661
* Add a regression testcase for #5674 #5677
* Fix C++ RPC build problem on Linux #5671
* Add a check Callback to the Pattern Paritioner #5646
* Call previous excepthook in `tvm_excepthook`. #5675
* Fix the shift column for `scale_shift_nchw` and `scale_shift_nhwc` in C topi #5679
* Support more dtypes for TVMDSOOp #5694
* In `memory_plan`, check if value is not None, instead of just checking value as boolean. #5700
* Fix flaky `test_topi_pooling.py:test_adaptive_pool` #5736
* Fix the values for `test_fmod` since it fails way too often otherwise #5723
* fix small bug about `dense_grad` #5695
* Fix sequential cpp test #5745
* Add Scatter to Topi/Relay/ONNX via hybrid script #5619
* Clean WASM environment before build #5759
* Fix gelu in PyTorch frontend, tighten numerical checks #5763
* fix #5686: remove a overstrict assert in MakeAllreduce (#5686) #5785
* Improve Pattern Language Docs #5676
* Add missing expr visitor for any #6082
* Remove the tvm web from version update #6122
* Clear relay cache after every build & Clear warning message cache after autotvm task extraction #6131
* avoid unexpected throw in AttrInitEntry #6128
* Verify that tensor reshape is valid. #6215
* Use LocalRunner by default in the tutorial tune_relay_cuda.py #6001
* Undefined names: import os for line 324 & import re for line 308 #6003
* GitHub Actions upgrade to actions/setup-python@v2 #6002
* Only pass pythonpath for ci images #6005
* Auto-convert shuffle with single index to “extract element” #6006
* Cache object refs in loop partitioner instead of object pointers #6004
* Fix `test_arith_solve_linear_inequality.py::test_multi_equal` #6014
* MXNet frontend support for AMP cast op #5976
* Demo showing how to run a pruned model. #5975
* Move compiler related registry items to `vta/build_module.py` #6012
* Pin keras version #6032
* Fix in `arm_cpu/conv2d_alter_op` for NHWC quantized #6027
* Add creation of Hexagon device in RPC client #6035
* Terminate basic block after “ret” instruction #6036
* µTVM CRT modifications for on-device RPC server #5921
* Create TBAA information based on the unrelying buffer type #6046
* Add support for tflite `arg_min` and `arg_max` #5992
* Fix `fully_connected` converter when batch size is not 1 #6038
* Fix a primitive check error #5991
* Refactor to expose MakeOp functions to C++ #6047
* Fix `conv2_gemm` after target structure update #6037
* Remove use of designated initializers from `hexagon_module.cc` #6055
* Build crttest and cpptest separately. #6057
* Fix pytorch frontend prim::Constant issue #6051
* update frontend tutorials to new model based runtime interface #6063
* Remove unnecessary std::cout #6072
* Fix error message in Buffer::vstore, NFC #6056
* Fix FSIM Compile Error. #6070
* Improve vector simplification for float operands #6043
* Fix LocalBuilder on macOS with python 3.8. #6083
* Add missing test for fast erf #6058
* Fixed point multiplication improvements for AArch64 #5980
* Fix code generation bugs for C/CUDA & Improve VerifyGPUCode pass #6041
* Delete declaration of unused `op_node` #6102
* Load configs even it has no entity #6100
* Update SGX example Cargo.toml #6067
* Add default value for option `USE_DNNL_CODEGEN` in the cmake #6099
* Update installation doc with minor improvements #6104
* lint: add opencl .cl file type #6092
* Clean up conversions between TVM and Rust functions #6114
* Improve reduction schedule on arm CPUs #6110
* Register Shape Func for Some Operators to Handle Dynamic Shapes #5955
* Fix variable name conflict with OpenCL keyword #6048
* Some rust cleanups #6116
* Option to specify alternate directory to output build to #6016
* Add `get_num_inputs` to GraphRuntime #6118
* TFLite quantized conv test #6084
* Fix autotvm on the `conv2d_nchw_winograd.mali` operator #6130
* add attr option mfloat-abi for arm32 #6123
* Fix CUDA Library Tuning #6132
* Add missing RPC sources after refactor #6113
* Correct `runtime.load_module` #6161
* Improve error messages in graph tuner, graph runtime, and module loader. #6148
* Fix some shape mismatches between TF and Relay #6166
* Improve doc string #6176
* Fix incorrect function signature in header #6172
* Fix alignment of note #6181
* Implemented PADV2 Operator for TFLite and added support for constant values in PAD. #6167
* Unary ops support added in frontend #6196
* Change the meaning of `conv3d_transpose` `output_padding` to match `conv{1,2}d_transpose` #6065
* Fix compile warnings. #6204
* Fix -mfloat-abi=soft compilation for ARM with OpenCL target #6150
* Match pytorch 1.6 googlenet pretrained model (#6201) #6212
* Mod operator, bug fix #6160
* RESHAPE with dynamic shape arg in TFLite frontend #6208
* Fix compilation error with cuda 11 #6213
* Fix `port_end` wrong default value 9199 to 9099 for keeping same with source code #6220
* Std op without specified dimensions support #6226
* fix crt building and running error #6231
* Implemented `ONE_HOT` Operator for TFLite. #6223)
* Avoid unexpected throw in AttrInitEntry #6128
* Added casting to hybrid script doc and fixed pass infra doc #6174
* Fix compile warnings. #6204
* Fix -mfloat-abi=soft compilation for ARM with OpenCL target #6150
* Mod operator, bug fix #6160
* Fix compilation error with cuda 11 #6213
* Fix `port_end` wrong default value 9199 to 9099 for keeping same with source code #6220
* Std op without specified dimensions support #6226
* Verify that tensor reshape is valid. #6215
* Fix crt building and running error #6231
* Fix `conv2d_transpose` output padding #6236
* Fix cuda half math function is undefined: hpow, htanh #6225
* Fix division range estimation error in simplifier #6244
* Fix newer GCC compiler warnings. #6257
* Support `_contrib_SyncBatchNorm` #6245
* Fix reduction #6250
* Add apt repository for clang-11 and llvm-11 #6256
* Update tutorial to new TARGET as `micro_dev` is no more #6262
* Fix clang-format #6264
* Trivial fix, up the rodata section for the discovery board to 512 bytes. #6259
* Fix cuda half math function is undefined: hpow, htanh #6253
* Add dilation in x86 NCHWc depthwise conv support #6267
* Decrease test times by introducing testing model #6235
* Add support for parsing the any dimension. #6277
* Improve error messages for memory verifier and gpu memory verifier #6281
* Reflect Compile-Time CMake Options into libtvm.so #6280
* Add cmake options into libinfo #6286
* Update slice to infer attributes when not graph inputs #6276
* Use rpc.LocalSession for simple tests #6294
* Fix random fail #6312
* Fix resize test #6298
* Fix cython FFI compact with np.int64 #6321
* Fix relay vm optimize #6322
* Changed TVMCTVMContext to TVMContext #6306
* Make able to compile with MSVC #6341
* ROCm changed name of library and removed the old one in ROCm 3.7 release. #6345
* Compatible for ROCm before 3.7 #6359
* Use clear name that is separate from ASF brand for cache #6360
* Fix `Dockerfile.demo_android` #6361
* Fx sparse dense schedule on cuda #5803
* Fix strategy for sparse dense cuda #5782
* Fix x86 conv2d template when tuning with unpacked layout #5938
* Fix the filter width parameter in `depthwise_conv2d` #6081
* Fix reshape usage in ARM schedule #5732
* Missing header #4865
* Fix `conv2d_transpose` output padding #6236
* Simplify reduce expression in te.gradient #6611

### API Changes
* `tvm.module` -> `tvm.runtime.module`
* `tvm.module.load` -> `tvm.runtime.load_module`
* `tvm.module.enabled` -> `tvm.runtime.enabled`
* `tvm.module.system_lib` -> `tvm.runtime.system_lib`
* `tvm.relay.Module` -> `tvm.IRModule`
* `tvm.create_schedule` -> `tvm.te.create_schedule`
* `tvm.placeholder` -> `tvm.te.placeholder`
* `tvm.compute` -> `tvm.te.compute`

### Deprecation
* Deprecate NNVM (#4535, #4562, #4565, #4571)
* Deprecate FreeStmt #5890
* Remove legacy `compute_expr.h` #5738
* Deprecate OpenGL #5711, #5712

## 0.6

### Relay in Production
Relay is a functional, differentiable programming language designed to be an expressive intermediate representation for machine learning systems. Relay supports algebraic data types, closures, control flow, and recursion, allowing it to directly represent more complex models than computation graph-based IRs (e.g., NNVM) can. In TVM v0.6, Relay is in stable phase and is ready for production.

* Algebraic Data Types (ADT) support (#2442, #2575). ADT provides an expressive, efficient, and safe way to realize recursive computation (e.g., RNN). Refer to https://tvm.apache.org/docs/langref/relay_adt.html for more information.
* Pass manager for Relay (#2546, #3226, #3234, #3191)
* Most frameworks have been supported in Relay, including ONNX, Keras, Tensorflow, Caffe2, CoreML, NNVMv1, MXNet (#2246).
* Explicitly manifest memory and tensor allocations in Relay. (#3560)

### Relay Virtual Machine
The Relay Virtual Machine (Relay VM) is the new generation of runtime to strike a balance between performance and flexibility when deploying and executing Relay programs. Previously, the graph runtime is able to utilize the fully static nature of the input graphs to perform aggressive optimization such as fully static allocation, and optimal memory reuse. When we introduce models which make use of control-flow, recursion, dynamic shapes, dynamic allocation we must change how execution works.

Relay VM is now usable and is able to achieve decent performance for a various of models and targets.

* Design (#2810 #2915) and a first version of implementation (#2889),
* Add VM runtime for Relay and compiler support (#3120, #3121, #2889, #3139)
* Relay VM (pattern matching #3470, port to python #3391, serialization #3647)
* Relay VM Profiler (#3727)
* Support execution on devices for Relay VM (#3678)
* [Relay][VM] Add more passes to VMCompiler (#4058)
* [relay][vm] Separate VM runtime with executable (#4100)
* Port VM, VM compiler, and Object into Python (#3391)
* VM: Add AllocTensor instruction and better instruction printer (#3306)
* [Relay][VM][Interpreter] Enable first-class constructors in VM and interpreter via eta expansion. (#4218)
* [Relay][VM] Clean up the VM and VM profiler code (#4391)

### Training
Relay is designed to natively support first-order and higher-order differentiation. The automatic differentiation infrastructure is now usable and a count of operators with gradient support are available in v0.6 release.

* Higher order reverse mode automatic differentiation that work with control flow (#2496)
* Higher order continuation passing style (#3456, #3485 )
* Relay gradient registration (clip #3509, `max_pool2d` and `avg_pool2d` #3601)
* Relay AD algorithm (#3585)
* Relay Training - allow gradient to return a tuple (#3600), numerical gradient check (#3630)
* Improve AD for concatenate (#3729)
* [Relay][Training] Add missing gradient check to gradient pass (#4169)
* As a part of Relay's automatic differentiation system, we are adding primal gradients for Relay operators. Please refer to #2562 for tracking the progress.
* Gradient for Conv2d (#3636)
* Add gradient operators (#3857, #3894, #3901, #3915)
* Add gradient for log-softmax (#4069)
* [Relay][Training] Add gradient for Crossentropy (#3925)
* [Relay][Training] Add and fix gradients (#4126)

### Quantization

Low-bit inference is getting more and more popular as it benefits both the performance and storage usage. TVM now supports two types of quantization. 1. Automatic quantizaion takes floating-point precision model, does per-layer calibration and generates low-bit model. 2. TVM also imports pre-quantized model from Tensorflow and MXNet, a new dialect QNN is introduced to handle further lowering to normal operators.

* Automatic Quantization
  - Low-bit automatic quantization supported. (#2116). The workflow includes annotation, calibration and transformation.
  - Refactor quantization codebase and fix model accuracy. (#3543)
  - KL-divergence-based per-layer calibration. (#3538)
  - Add option to select which convolution layers are quantized. (#3173)
  - [Relay][Quantize] Integrate data-aware calibration into quantization. (#4295)
* Pre-quantized model support (QNN operators and legalize pass).
  - Add a legalize pass to Relay (#3672)
  - Qnn Concatenate, quantize, dequantize and requantize operators (#3819,  #3730, #3745, #3531)
  - QNNtoRelay & QNNLegalize Pass utility (#3838, #3782)
  - Requantize: Optimize lowering for some corner cases. (#3864)
  - New quantized operator support: conv2d, add, dense (#3580, #3736, #3896, #3910)
  - Do type checking for the input and kernel in the qnn conv2d (#3904)
  - Legalize and AlterOpLayout for Intel int8. (#3961)
  - Renaming tests to follow the Relay nomenclature. (#3975)
  - Fix padding changes due to #3739 (#3989)
  - Memorizing quantize node mapping to avoid duplicated simulated quantization (#3233)
  - Infrastructure to support pre-quantized models (QNN) (#3971).
  - [Relay][AlterOp] NHWC to NCHWc support for Pool, concatenate, sum. (#4059)
  - [TOPI][x86] Cascade lake support. (#4123)
  - [TOPI][x86] Legalize - Support int8xint8 convolution to use VNNI inst (#4196)
  - Qnn dequantize with min max using Mxnet flavor to support Mxnet prequantized models. (#3945)
  - Improve the lowering of Qnn Dense (#4213)
  - Adding support for dequantizing from int32 to float32. (#4130)
  - [QNN] Refactor fixed point multiplication in requantize (#4073)
  - [Relay][Quantize] Use fixed point mulplications (#4160)
  - Add support for quantized multiply to Relay (#4141)
  - Use legalize to handle NHWC layout for `arm_cpu` (#3754)
  - [QNN][Legalize] Specialize for Platforms w/o fast Int8 support (#4307)
  - [QNN] Use Int16 upcast in Fallback Conv2D. (#4329)
  - Retain input kernel scales in QNN dialect (#4292)
  - [QNN] Lowering for Depthwise Convolution. (#4351)
  - [QNN][TFLite] Parsing QNN Add op. Adding MobilenetV2. (#4142)
  - [QNN][TFLite] Parsing TFLite quantized models. (#3900)
  - Added tflite frontend support for quantized mean. (#4339)
  - [Relay][Legalize] Legalize `conv2d_transpose` for NHWC (#4399)

### Accelerator and Microcontroller Support

TSIM is introduced to improve software and hardware integration and simulation accuracy. It integrates the hardware development process into the software stack. TSIM enables VTA to provide a more accurate performance feedback, i.e. clock cycles, compared to the traditional functional model of a hardware accelerator. Moreover, Chisel implementation for VTA is availale and it runs on top of TSIM.

There has been a proliferation of resource-constrained and embedded devices that do not have operating systems or a mature software stack. MicroTVM is intended to support TVM on such bare-metal devices.

* [TSIM] Enabling Cycle-Accurate Hardware Simulation for VTA (#3010, #3206, #3242)
* Chisel implementation for VTA and runs on top of TSIM (#3258, #3347)
* MicroTVM (#3227)
* Relay Compilation + AutoTVM compatible operator libraries for VTA (#3135)
* ChangeBatch pass for batched VTA compilation (#3656, #3660)
* VTA fast simulator statistics (#3481)
* TSIM improvements and fixes (#3505)
* Chisel VTA enhancements and fixes (32bit support #3558, alu instruction generation #3592, coherence support #3593, separate types #3605, tensor issue/commit #3637, uop load request #3643, uop dma requests #3654)
* VTA Runtime refactor for non-shared memory FPGAs (#3590)
* VTA HLS codebase refactor for Ultra96 (#3496)
* VTA support for batched inference (#3661)
* VTA bitstream compilation for Intel FPGA (#3494)
* TSIM: Introduce Virtual Memory for TSIM Driver (#3686)
* Parallel TSIM hardware compilation with macOS and debug support (#3797)
* Chisel: scale dram base address in hardware instead of runtime (#3772)
* Chisel: run all unittests by default (#3766)
* Chisel: improved Data Gen, Added ALU Test (#3743)
* Chisel dependencies for TSIM CI (#3721)
* Chisel: Added Module Unit Test Infrastructure (#3698)
* Add ISA BitPat generation (#3891)
* de10-nano driver (#3394)
* Extending Vision model coverage compilation for VTA (#3740)
* Conv2d transpose (deconvolution) operator support (#3777)
* Support TLPP in function simulator. (#3555)
* [VTA][Chisel] TSIM VTA Source Refactor (#4163)
* [VTA][TSIM] Serial GEMM Application Added (#4082)

### Rust Support
Rust language support in TVM includes two parts. 1. The frontend wraps the current C API and exposes a Rust programming model. 2. The backend serves as an alternative to C++ runtime. It privdes a standalone WASM module and security support, e.g., SGX.

* Rust frontend (#2292).
* Unify types between bindings and pure Rust impl (#2616)
* Rust: load syslib modules at compile time (#3274)
* Rustify PackedFunc & Friends (#2969)
* Rust DSO module (#2976)

### Operator Support
* A special operator `annotation.stop_fusion` to prevent it being fused with previous expressions (#2624).
* `batch_matmul`  supported (#2561).
* `reverse_reshape` supported (#2503).
* Faster-RCNN proposal operator for CUDA (#2420).
* Vision operator for YOLO `yolo_reorg` (#1941).
* `slice` operator for MXNet (#2662).
* `arange` supported (#2621).
* Vision operator `roi_align` (#2618).
* `where` operator for MXNet (#2647).
* Deformable conv2d (#2908)
* Faster-RCNN Proposal OP (#2725)
* ROI Pool operator (#2811)
* Gluoncv SSD support on CPU (#2353)
* shape, reverse, and sign op (#2749, #2800, #2775)
* tile and repeat op (#2720)
* logical operators (#2743, #2453)
* stack op (#2729)
* NCHWc upsampling (#2806)
* clip and wrap mode support in take (#2858)
* AlterLayout support for `intel_graphics` conv2d , depthwise conv2d (#2729, #2806)
* Add foldr1 operator (#2928)
* Add rsqrt operator (#2949)
* Add clip and wrap mode support in take (#2858)
* `Gather_nd` exposed to relay (#2945)
* `bitserial_conv2d` move to autotvm template and updates (#2819)
* Port x86 NCHWc to AutoTVM for Task Extraction (#2664)
* Implement relay `nn.bias_add` compute in C++ (#3027)
* Rename output tensors for better readability (#3006)
* int8 dense on CUDA & Dense op quantization (#2877)
* Bitserial dense operators for CPU (#3051)
* Enhance upsample operator to adapt onnx opset v9 (#2968)
* Add adaptive pooling operator (#3085)
* Add all operator (#3124)
* Add cblas `batch_matmul` (#3210)
* Add packing for int8 1x1 convolution and support the int8 group convolution on X86 (#2991)
* Add op size (#3094)
* x86 TOPI (`roi_align` #3475, `conv2d_transpose` #3491)
* Intel INT8 (dilation in conv2d #3510, type checking #3516)
* Reinterpretation of tensor elements (#3599)
* Spase-Dense for block-sparse multiplication (#3566)
* Winograd matrix computation (#3553)
* CUDA schedule for `pool_grad` (#3622), `group_conv2d` (#3663)
* Bitserial operations conv2d, dense and bitpack (#3844)
* Improve numeric gradient check (#3856)
* Resize rework ([3788](#3788))
* Improve `conv2d_transpose` CUDA schedule template (#3796)
* SpaceToDepth and MirrorPad Operators (#3718)
* Add variance and layer norm op (#3700)
* Add `sparse_transpose` for Square CSR matrices (#3707)
* TOPI: Memoize winograd matrix (#3687)
* New TOPI operators: `erf`, `logical_and`, `logical_or`, `logical_not`, `isnan` (#3702, #3929, #3979)
* Improve `ceil_divide` in tile/split (#3842)
* [Relay][Frontend][TF] Add tensor array ops (#3798, #4309)
* [TF][Op] Op where (#4045)
* [TOPI]Add op argwhere (#3994)
* [Relay] `crossentropy_with_logits` and its gradient (#4075)
* [Relay][Op] Enhance Upsample Operator to support float scales (#4206)
* [Relay][Op] Add instance norm op (#4004)

### Frontend and User Interface
* Frontend darknet (#2773)
* Support tf.gather (#2935)
* Support tf.where (#2936)
* Adding ADD operator to tflite frontend for compiling the MobileNetV2 (#2919)
* Support SpaceToBatchND/BatchToSpaceND in Tensorflow frontend (#2943)
* Simplify TF `get_output_names` (#3025)
* TF Tile Round Sign Pow Exp Reverse (#2960)
* Gluncv SSD support on the GPU (#2784)
* Allow an op as loop var in Tensorflow (#3056)
* Add `FULLY_CONNECTED` op into tflite frontend (#3019)
* Add MXNet converter for RNN layer ops (#3125)
* Add log op in tf frontend (#3111)
* Add SoftPlus Sqrt in Tensorflow frontend (#3187)
* Add onnx elemwise greater/less (#3186)
* Add PlaceholderWithDefault (limited) implementation in TensorFlow (#3184)
* Support `tf.math.reduce_prod` (#3166)
* Better shape inference in TensorFlow Frontend (#3176)
* Get list of unsupported ONNX operators (#2995)
* Implement ONNX MaxPool-v8 and MaxPool-v10 (#3114)
* Convert TFLite NCHW to NHWC (#3141)
* Add Crop op converter (#3241)
* TFLite frontend operator support: PAD, RESIZE, MUL, Reduce (min, max, mean, prod), LOGISTIC, elemwise operators (Sub, Divide, Power, Max, Min) (#3310, #3370, #3304, #3421, #3313, #3357)
* Tensorflow frontend operator support: Abs, FloorDiv, GatherND, LeftShift, LogSoftmax, Max, Min, Mod, RightShift, ZerosLike, TruncateMod, Neg, ClipByValue, ResizeNearestNeighbor (#3270, #3211, #3393)
* TFLite: Add `fused_activation_function` for ADD, SUB, MUL, DIV (#3372)
* Support bidirectional RNN layer for MXNet (#3397)
* TFLite operator support (pack #3521, split #3520 )
* Keras operator support (permute, softmax #3618)
* TF operator support (BatchMatMul #3634)
* TFLite frontend operator support: tile, transpose (#3814, #3705)
* ONNX frontend operator support: PReLU for NNVM, Not, Sign, Equal (#3813, #3836, #3760)
* Keras frontend operator support: Dot (#3668)
* Add more cases to Keras `_convert_reshape` (#3846)
* TensorFlow frontend operator support: OneHot, log1p, cos, sin (#3781, #3614)
* Support BatchMatMul with input dimensions larger than 3 for TensorFlow (#3732)
* ONNX new operator support: And, Tile, Erf (#3878, #3941, #3988)
* MXNet new operator support: pad, conv1d, deconv1d (#3739)
* TFLite new operator support: `batch_to_space_nd`, `space_to_batch_nd`, tanh, greater, relu (#3850, #3996, #3963, #4022)
* TFLite: Support depthwise convolution multiplier greater than 1 (#3922)
* Keras: Fix ReLU in Keras Converter missed the case (#3917)
* Keras: frontend upsample and 1 channel conv2d fixes (#3937)
* Tensorflow: Convert scalar Const into tvm.relay.const (#3885)
* TensorFlow: Add support for SquaredDifference (#3930)
* [relay][frontend] clean up tf frontend (#3710)
* [Relay][Topi][TensorFlow][ONNX][Lang] Add support for Any op (#4205)
* [Relay][Frontend][ONNX] Add support for op Where (#4184)
* [Relay][TopHub] Add switch to disable TopHub download (#4015)
* Add parser support for CAST tflite operator (#4096)
* Add parses support for `zeros_like` tflite operator (#4042)
* Add parser support for SUM tflite operator (#4182)
* Add support for tf.assert (as no-op) and `tf.no_op` to TF Relay frontend. (#4172)
* [Relay][Frontend][ONNX] New Operators and Opsets to Support BERT (#4197)
* [Relay][Params] Add APIs for storing and retrieving parameters from individual functions. (#4194)
* Add `build_create_shared_func` to tvm/contrib/cc.py (#3840)
* Tensorflow saved model for NNVM ([#2493](#2493/) and Relay ([#2586](#2586/)).
* Introduced `HybridModule` (#2477) so that normal TVM schedule can be compiled to hybrid target, run and dumped to Hybrid Script.
* Relay ][Frontend][Tensorflow] add operator `add_n` (#4181)
* [Relay][Frontend][Tensorflow] StopGradient (#4238)
* [Relay][Frontend][ONNX] Add support for broadcasting to Where and MatMul (#4267)
* [TFLite] Support PRelu (#4298)
* [Frontend][MxNet] support mxnet cond op (#4311)
* Add support for `quant.mul` operator in tflite frontend (#4283)
* [Relay][Frontend][ONNX] operator support: DepthToSpace, SpaceToDepth (#4271)
* [Relay][Frontend][Tensorflow]Add `conv2d_transpose`. (#4300)
* [Frontend]Add TensorFlow FloorMod (#4308)

### Runtime and Backend Support
* Make external library extend TVM's NDArray more easily (#2613).
* Improvements for NNPACK integratation, includes ci test, winograd (#2846, #2868, #2856, #2721)
* Improvements for OpenCL runtime (#2741, #2737)
* GraphRuntime: Enable sharing parameters of a model among multiple threads (#3384)
* Android runtime argsort support (#3472)
* GraphRuntime enhancements (`set_input_zero_copy` #3416)
* A new minimal runtime implementation (~12kb .text on ARMv7/x86) for TVM.
* Add AVX512VNNI support for TVM (#3388)
* Enable miopen Group Convolution (#3987)
* Minimal runtime (~12kb .text on ARMv7/x86) for subset of TVM models (#3567)
* [RUNTIME] Separate runtime related contrib into runtime/contrib (#4207)
* [topi] add ARM v8.2 udot (uint8) support (#3978)
* [codegen] Add multiple operands and function support when using fp16 compilation (#4056)
* [TOPI] Added support for Mali Bifrost target (#4047)
* [topi] enable fp16 sort for arm (#4084)
* Add OpenOCD Low-Level Device (RISC-V Support) (#3756)
* Add wave 32 bc for AMD ROCm backend (#3984)
* [RUNTIME] Support C++ RPC (#4281)
* [TOPI][OP] Support Faster-RCNN Proposal OP on CPU (#4297)
* [TVM][RUNTIME] A minimum example to generate external library wrappers for DSOModule (#4280)

### Language and Architecture
* Support custom datatypes (#2900)
* Add the acc16 intrinsic support (#3081)
* Handle float16 constants & fix BatchNorm (#3260)
* Structural hash - incorporate the var type into its hash (#3267)
* Relay C++ Build Module (#3082, #3144, #3174)
* Enable decorating python class to be a Relay Pass (#3364)
* Make Partial Eval support interprocedural optimization and termination check. (#3033)
* Introduce feature manager to Relay. (#3236)
* Use Relay parser to define the Relay prelude (#3043)
* Mechanism to detect incomplete expression match in Relay (#3203)
* EQ/NE operators support for StringImm expressions (#3283)
* Mechanism to detect incomplete expression match in Relay (#3203)
* Introduce CanonicalizeCast pass to formally reduce memory overhead introduced by fused cast operations (#3280)
* Support overloading comparison operations in Relay (#3168)
* Mac count: provide a pass to calculate the number of multiply-accumulate operations in a network (#2609).
  - support for `conv_2d_transpose` (#3469)
  - [Relay][Pass] Count MAC for BatchMatMul (#4157)
  - Detect depthwise conv2d in `mac_count` pass (#3083)
* Add Tuple pattern (#3596)
* Text format support for ADTs and prelude (#3863, #3939)
* Add new IR pass CombineParallelDense (#3862)
* Add support for `EQ` op in the deduce bound and the loop partition (#3775)
* Introduce base-class IRMutatorWithAnalyzer (#3969)
* Define more standard global functions in the prelude of relay program, includes foldr1, hd, tl, nth, list update (#2928, #2917, #2771, #2866)
* Add SkipVectorize pass (#3222, #3228)
* [Relay][Pass] Add pass to remove unused functions in relay module (#4334)

### Symbolic shape enhancement
* Add shape function for symbolic shape. It enables certain cases for broadcast with symbolic shapes. (#3606)
* [tvm][any] broadcast with values other than one (#3967)
* Symbolic shape support (broadcast op #3389)
* Support reshape for dynamic shape in tf converter (#4185)
* Runtime Shape Functions (#4179)

### Language and Architecture
* An optimization pass to eliminate expressions which have the same functionality and same inputs (#2639).
* Refactor text printer to add stream-like API and FunctionType support (#2605, #2882)
* Build a scaffold for structured error handling (#2838). The new mechanism detects and rewrites error messages so that c++ and python stack trace are unified and not redundant. Guideslines and conventions for error handling is also discussed.
* Higher order reverse mode automatic differentiation that work with control flow (#2496)
* Integer arithmetic analyzers, includes modular set analysis, const integer bound analysis and rewrite simplifier (#2904, #2851, #2768, #2722, #2668, #2860)
* Improve operator fusion for TupleGetItem in relay (#2914, #2929
* Compute FLOP of autotvm template for int8 models (#2776)
* Common subexpression elimination pass in Relay (#2639)
* Improve quantization in Relay (#2723)
* Refactor `build_func` in measure module of autotvm to better support cross compiler (#2927)
* Quantize all fields of concatenate (#2913)
* Remove stale verilog generator (#2964)
* Improve Relay printing (#2984, #2881, #3030, #3041)
* Add `min_num_branches` option in CombineParallelConv2D (#2961)
* Add `expr_visitor`, fix `expr_functor` exponential blowup problem (#2988)
* Support Deriving channels when it is not provided in AlterLayout. (#2972)
* Enhance BoundDeduce algorithm (#2795)
* Enhance loop partition algorithm (#2956)
* Better tuple fusion implementation (#3092)
* Enhance fusion rule that starts from elemwise and broadcast (#2932)
* Remove `on_device` op after annotation in heterogeneous pass (#3204)
* Improve canonical and rewrite simplifier (#3132, #3149)
* Capture constant external python variables in hybrid script (#3157)
* Remove Peano nats from the prelude (#3045)
* Macro to define NodeRef methods, constructor style example (#3224)
* Consistent RAII scoping API (#3231)
* Register all operators' attributes in Python (#3175)
* Add module supoort in relay.build (#3424)
* Relay pass infrastructure improvement (#3319, #3336, #3430, #3353)
* Migrate Relay passes to pass manager (#3323, #3289, #3251, #3406)
* Improve heterogeneous annotation by using visitor (#3261)
* Support export ADT value in Python (#3299)
* Extend TensorComputeOp to allow scalar inputs (#3300)
* Transitioning low-level IR away from HalideIR (#3533, #3535)
* Tags for ADT constructors (#3369)
* IR dumping for debugging (#3493)
* Pretty printer and parser roundtrip (#3460, #3536)
* Relay type checking (conv2d weight dimension #3511, any shape #3221)
* Relay Module enhancements (remove free variables #3476)
* LLVM DWARF debug information (#3420)
* Printer for Layout/BijectiveLayout (#3582)
* Type inference escape hatch (#3571)
* Making iterators compatible with constructors of STL containers (#3624)
* Moving Conv, Dense, Concatenate InferTypes to header (#3783)
* Simplify casts of constants 0 and 1 (#3758)
* Conditionally replace reduction init axis. (#3408)
* Improve Partial Evaluator (#3749, #3703)
* Strict mode in Relay pattern matching (#3620)
* Quit and clean when TVM is interrupted (#3640)
* Make Type Relation catch more errors (#3899, #3699)
* Refactor the way we interface between different modules of Relay (#3906)
* Introduce `schedule_injective_from_existing` and unify external schedules for all targets (#3983)
* [NODE][REFACTOR] Refactor reflection system in node. (#4189)
* Unify node system and object (#4161, #4115, #4128)
* [Relay][Refactor] Rename Datatype to ADT (#4156)
* [Relay] fix exponential blowup in interpreter (#3559)
* [Relay] Fix memory leak in the interpreter (#4155)
* [rpc] use callback func to do send & recv (#4147)
* Add `lift_if_then_else` pass to improve loop partitioning (#3865)
* Decrease the complexity of CalcDep from exponential to linear (#4053)
* [IR] Make iterators compatible with constructors of STL containers (#3624)
* [Relay][Pass] Avoid FoldConstant folding some ops (#4245)
* [Relay][Prelude] More dtypes support in `tensor_t` (#4233)
* [NODE][REFACTOR] Rename IRFunctor->NodeFunctor, use func pointer (#4247)
* [RUNTIME][REFACTOR] Use object protocol to support runtime::Module (#4289)
* [CodeGen] Add build config option `disable_assert` to control whether to generate assert. (#4340)

### Arithmetic Analysis
* Formalize Integer Arithmetic Analysis (RFC: #2588). It is aiming to perform better context-dependent analysis, bound analysis, centralized arithmetic logic and arithmetic simplification. (#3272, #3463, #3464, #3368, #3503, #3504 , #3502, #3479 , #3568)
* Introduce FloorDiv/Mod, TruncDiv/Mod, and IndexDiv/Mod for better arithmetic simplification (#3976, #3986, #4000, #4014, #4008, #4028)
* [ARITH] Use floordiv for the deduce bound (#4025)
* [Simplifier] Rewrite simplification rule to eliminate unnecessary conditionals. (#4076)

### Runtime and Backend Support
* Provide error msg for failure function call in tvm4j (#2967)
* Expose backtrace symbols in Debug mode (#3001)
* C++ GraphRuntimeCodegen, Deprecate Python2 (#2986)
* Ensure interpreted functions can take values that are not TensorValues (#3015)
* Make OpenCL runtime Compatible with OpenCL2.0 (#2897)
* Handle INF and NAN in CUDA and OpenCL (#3194)
* Update debug graph runtime for more precise layerwise timing (#3232)
* ROCM support (llvm printing #3662, ld.lld finding #3664, save to file #3665)
* Threadpool: make `spin_count` configurable (#3577)
* RPC worker children termination (#3669)
* Vulkan runtime reimplementation (stream approach) (#3849)
* Vulkan backend supports Call::reinterpret and vectorized comparison (#3795)
* Support MKL on Windows (#3837)
* Vulkan IR builder (bool to float #3513)
* Force `code_object_v2` for amd gpu backend (#4099)
* [Codegen][cuda-fp16] fallback to fp32 simulation when cuda arch < sm53 (#4268)
* Fix and refactoring for AMD gpu backend (#4305, #4321, #4341, #4342)
* [Debugger] Sorting op-time breakdown for quicker analysis. (#4352)
* [nvcc] enable multiple arch in one fatbin (#4377)
* [RUNTIME] Move module export to the function level. (#4405)


### Frontend and User Interface
* Relay now supports saving and loading parameter dictionaries. (#2620)
* Add `max_num_threads` to Hybrid Script, which allows users to get max number of threads for GPU targets ([#2672](#2672/)).
* Improvements for tensorflow frontend (#2830, #2757, #2586), includes decompiling tf control flow (#2830)
* Improvements for mxnet frontend (#2844, #2777, #2772, #2706, #2704, #2709,, #2739)
* Improvements for keras frontend (#2842, #2854)
* Improvements for DarkNet frontend (#2673)
* Improvements for ONNX frontend (#2843, #2840)
* Better profile result dump in Chrome Tracing format (#2922, #2863)
* Unified error handling in NNVM and Relay frontends (#2828)
* Improve NNVM to Relay conversion (#2734)
* Remove `input_0d_mismatch` special handling for TF Frontend(#3087)
* Bumped ONNX version from 1.1.0 to 1.4.1 (#3286)
* Simplify parameter handling in Tensorflow frontend (#2993)
* CoreML improvement for image scaler and padding (#3800)
* Clean up TensorFlow frontend (#3710)
* Darknet: Solve tvm parsing darknet resnext failure bug (#3778)
* Frontend changes `get_workload` - (#3483)
* [TF][Relay][Op] Pass module when infer shape (#4287)

### AutoTVM
* Support override in `register_topi_compute` and `register_topi_schedule`. (#3292)
* Improve graph tuner dealing with Tuple. (#3649)
* Add AutoTVM template for conv2d Intel int8. (#3955)
* Add AutoTVM template for dense on CUDA. (#3923)
* Add AutoTVM template for conv2d on Intel graphics. (#3839)
* Optimizing autotvm task extraction speed. (#4138)
* [AutoTVM] Add `batch_matmul` to tunable operations. (#4242)
* Selecting tuning templates when extracting task. (#4338)

### Performance Improvements
* Enable AlterOpLayout pass for x86 on Relay (#2585). It is essential to get decent performance for CNN-based model on Intel CPUs.
* Better intrinsic matching for x86 CPU and ARM CPU, includes variants of vcvtph2ps and vmlal.s16 (#2925, #2748).
* Improve injective schedule for ARM CPU(#2801)
* Core functionality for Graph tuner (#2184)
* Fast tanh implementation (#3255)
* Improve multi-batch conv2d on x86 (#3308)
* Improve `non_max_suppression` and `get_valid_counts` for CPU (#3305)
* Improve `roi_align` performance for CPU (#3296)
* Improve `nms` and `get_valid_count` performance (#3282)
* Graph tuner for multiple subgraph (#3490)
* For sparsity, fast transpose for square CSR matrices has been now merged, which is a good start point for more general sparse type support.
* Reduce `set_input` and `set_input_zero_copy` overhead (#3805)
* Parallelize batch axis for ARM (#3931)
* Support cuBLAS BatchMatMul (#3936)
* Add AVX512VNNI support for TVM (#3388)
* Enhance tuning space of split (#3949)
* Enable miopen transpose convolution and fp16 support (#3952)
* Improve `conv2d_transpose` schedule on X86 and CUDA (#3948)
* Expose llvm.nearbyint intrinsic (#4001)
* [TOPI][X86] Pool operator parallel support. (#4090)
* Improve layout for several operators (#4103, #4040, #4080)
* [Relay][VM] Fix constant folding issue in VM compiler (#4077)
* [relay][vm] Reuse allocated device memory (#4170)
* [Runtime] Enable option to use OpenMP thread pool (#4089)
* [PERF] Parallelize reduction for CPU (#4158)
* [TOPI] Tunable Template for Conv2D HWCN on CUDA (#4168)
* [TOPI] Add valid auto tvm for Intel Graphics (#4078)
* [TOPI] FIFO buffer op, to accelerate sequence modeling with dilated convolutions (#4039)
* TensorCore Support using Intrinsic (#4136)
* Auto TensorCore CodeGen (#4234)
* Use cblas for dense and `batch_matmul` (#3787)
* Update TOPI softmax compute and CPU schedule (#3680)
* [VTA] Performance optimize, remove unnecessary contigious memory use. (#4246)
* [TOPI][AlterOpLayout][ARM] Enabling NHWC to NCHW layout transformation. (#4249)
* [PERF] Parallelize reduction for CPU (#4158)
* [ThreadPool] Solve thread transitions issue (#4344)

### Documentation
* Tutorials for deep learning frameworks support in Relay.
* Tutorial for running AutoTVM with Relay (#2594).
* Document for Algebraic Data Types (#2575).
* Move NNVM tutorials to Relay (#2783, #2785, #2766, #2693)
* Documentation on operators (#2761)
* Add gradient operator tutorial docs (#2751)
* Add compiler pass tutorial docs (#2746)
* Add Android Tutorial (#2977)
* Developer documentation for InferBound pass (#3126)
* Add missing targets to `target_name` documentation (#3128)
* Various documentation improvements (#3133)
* Add VM doc (#3188)
* Update documents for TSim (#3409, #3318, #3302, #3343, #3206)
* Improve tvm4j document describing LLVM support (#3404)
* Tutorial migration to Python3 (#3498)
* Android RPC README (#3500)
* Documentation for Relay opcode (#3522)
* Tutorial for pass manager (#3515)
* Minimum version of Python in docs (#3588)
* Relay pass infra (#3583)
* X86 Autotune tutorial improvements (#3609)
* YOLOv3 tiny Darknet tutorial (#3674)
* SSD doc to avoid confusion (#3677)
* Tutorial: Build a Graph Convolutional Network on TVM (#3681)
* Add docs for analysis namespace (#3985)
* [tutorial] Relay pass infra tutorial (#4083)
* [DOCS] Add TensorFlow frontend docs (#4154)
* Tutorial: update Building a Graph Convolutional Network tutorial (#4060)
* [Docs] Add dependency of compilation with LLVM (#4117)
* [Documentation]Fix example code in comment of `tvm.build_module.build()` (#4195)
* TSIM: add virtual memory support to examples (#3868)
* Relay pass infra tutorial (#4083)
* Fix the TF tutorial to run against TF2.0 and TF1.x (#4104)
* Add `topi.nn.fifo_buffer` to TVM doc (#4343)
* License statement (#4345, #4359, #4401, #4402, #4408, #4409, #4410, #4414, #4431)

### Build and Test
* Increate the robuteness of CI test (#2841, #2798, #2793, #2788, #2781, #2727, #2710, #2711, #2923)
* Improve conda build (#2742)
* Add caffe2 nnvm frontend to CI (#3018)
* Use bridge network and expose port on macOS when launch docker image (#3086）
* Run DarkNet tests (#2673)
* Add file type check (#3116)
* Always run cpptest during build to ensure library correctness (#3147)
* Handle more file types in ASF header (#3235)
* Add `test_forward_ssd_mobilenet_v1` to `tflite/test_forward` (#3350)
* Add Azure build pipeline (#3458, #3459)
* Update ci-gpu to v0.52 (#3374)
* Enable more visible symbols by default (#3365)
* Separate out legacy as a stage in CI (#3337)
* Simplify build script, remove python 2 support  (#3419)
* Ignore rust cargo lock files in rat (#3314)
* Improve CUDA Conda package build (#3281)
* Update CMakeLists.txt to be more flexible to find the third parties libraries (#3354)
* Docker update conda package (#3344), requests and pillow (#3495), Android demo (#3499), rat install (#3527), ARM support (#3546), LLVM (#3590)
* Relay-to-Python testing (#3156)
* Code refactoring/remove (#3523, #3667)
* Zero-rank testing (#3612)
* CMake compilation (#3611, #3650, google test #3628)
* Standalone wheel build for TOPI (#3657)
* Fixing performance issues in PassUpDomain when fusing and splitting axes (#3073)
* conda recipe (#3791)
* Allow users to specify download directory (#3803)
* Update docs for installation for CUDA (#3832)
* Update `hybrid_script.rst` (#3799)
* Acknowledge Halide attributions (#3824)
* Add psutil dependency (#3780)
* Temporary disable rust test (#3809)
* Solve occasional CI issue when pad value is all 0 (#3801)
* Towards TSIM CI testing (#3704)
* Use pip3 for python3 (#3742)
* Update docker image `ci_cpu,i386` to include verilator (#3738)
* Remove sccache from Rust install (#3728)
* Update dmlc-core to the latest commit (#3716)
* Update GPU docker (#3709)
* Add an option to build with -pthread (#3671)
* Add DGL to `{ci_gpu, demo_cpu, demo_gpu}` docker images (#3692)
* Use pytest instead of nosetest (#3524)
* Enable NHWC of `relay.testing.mobilenet` (#3886)
* Add .hsaco save/load for `tesnor_expr` Tutorial (#3852)
* Support LLVM trunk (#3907)
* Remove GTest cmake flag from install docs (#3953)
* Allow `USE_LLVM` to take extra arguments (#3954)
* [CI] Pin NNPack pthreadtools version (#4152)
* [TOPI] Fix flaky testcase for check round (#4211)
* [CI] Move gpu docker binary to cuda10 (#4229)
* [CI] use llvm9 for the gpu tests (#4224)
* [CI] Update GPU docker to cuda10 (#4228)
* [Relay] Install Relay Prelude program in package install (#4227)
* [relay] use `time_evaluator` for measurement (#4191)
* [Relay] Improve build error when no lowered funcs are produced (#4132)
* [llvm] switch to use Align for llvm trunk (#4051)
* [CUDA] Update `have_int8` condition to run on compute capability 7.x devices (#4214)
* [DOCKER] Pin torchvision==0.4.1 (#4140)
* [DOCKER] torch install depends on future package (#4098)
* [CodeGen] Disable -mfloat-abi hard option for LLVM < 6.0 (#4071)
* Add a python how to example of deploying tvm module with tvm runtime only (#4094)
* Hide symbols from dependent libraries if `HIDE_PRIVATE_SYMBOLS` is ON. (#4041)
* [BUILD] Disable utvm standalone runtime by default (#4240)
* Fix TSIM compile error in Linux (add missing -fPIC flag) (#3876)
* Add scalafmt and format existing scala codebase (#3880)
* Update TFLite wheel version to 1.13.1 (#3435)
* Remove PEP498 f-string new feature for support python3.5 (#4250)
* Require LLVM >= 9 for AMDGPU backend (#4253)
* Rename ml.dmlc.tvm to org.apache.tvm (#4290)
* [Test][TF][Relay] Fix argument preparation for vm test mode (#4296)
* Add test for the `qnn_add` operator (#4282)
* [CI][DOCKER] Add ONNX runtime dep (#4314)
* [CI][DOCKER] Upgrade image to include onnx runtime (#4313)
* [CI] Set workspace to be per executor (#4336)
* [Build][Windows] Fix Windows build by including cctype (#4319)
* [Contrib] Add MKL DNN option (#4323)
* [Test][Relay][Pass] Add test case for lambda lift (#4317)
* Remove Python imp module as it is deprecated (#4275)
* Bump up CUDA log version in tophub.py (#4347)
* Add rule for clean in APPs (#4364)
* [Relay tests] Temporary Attr Update for Order-Independent Testing (#4357)
* [CI] Avoid content-length request in test data download (#4375)
* Compare all outputs in TFLite `test_forward_ssd_mobilenet_v1` (#4373)

### Bug Fixes
* [RELAY] Fix `get_int_tuple`. (#2691)
* [ARITH] Select support for integer set analysis. (#2687)
* [Relay] Fix error in ANF (too agressively inline atomic expression and create free variable). (#2665)
* [Hybrid Script] Fix name conflict and attached scope problem. (#2649)
* [Relay] Fix ANF for reference and pattern matching. (#2637)
* [Relay] Fix fusion bug when call symbol that is not an operator. (#2630)
* Fix missing <sstream> header file. (#2629)
* [Relay]Fix the bug in heterogeneous annotation which mistakenly steps into the fused op. (#2622)
* [AutoTVM] Fix incorrect localhost usage in RPC mode. (#2619)
* [NNVM] Fix incorrectly getting layout attribute as a tuple. (#2610)
* [Relay] Fix mutating IF expression. (#2601)
* [Tutorial] Fix downloaded file path. (#2590)
* [Storage] Fix int32 overflow bug when input is big. (#2580)
* [NNVM] Fix non-identity problem for FInplaceIdentity. (#2572)
* [Golang] Fix compilation error. (#2558)
* [Tensor Expression] Fix missing reduction init predicates. (#2495)
* [Relay] Fix missing argument for NCHWc in Relay. (#2627)
* [TOPI] Fix `Nms_ir` data race. (#2600)
* Fix `compute_inline` with multiple outputs (#2934)
* [TEXPR][PASS] Fix thread all reduce to avoid write after read hazzard (#2937)
* [FRONTEND][TENSORFLOW] bug fix for tensorflow official slim models. (#2864)
* [FRONTEND][ONNX] Some bug fixes and Shape operator fixed for relay. (#2850)
* Turn on `USE_SORT` by default (#2916)
* [DOCKER] Upgrade ci-cpu to latest v0.50 (#2901)
* [TESTS] Import script robustness (set -u) (#2896)
* [Relay] Fix name of bias in testing.mlp (#2892)
* [TESTS] Improve script robustness (#2893)
* Add dense schedules to `__init__` for cpu (#2855)
* [Apps] [howto_deploy] fix cxx-flags order and build directory (#2888)
* [Relay] Add TVM_DLL for ANF/GNF conversion #2883
* [Relay] Fix Relay ARM CPU depthwise spatial pack schedule alter op layout issue. (#2861)
* Fix setting up hints for getaddrinfo (#2872)
* Add missing sgx includes (#2878)
* Fix error reporting for missing axis (#2835)
* Fix an OrderDict initilization bug. (#2862)
* Fix Xcode 10 metal compile error (#2836)
* tvmrpc: Fix includes (#2825)
* Fix `init_proj.py`: Team ID expected (#2824)
* [DOCKER] Fix git clone failure. (#2816)
* upgrade java style-check due to CVE-2019-9658 (#2817)
* [Relay][Quantization] Fix duplicated simulated quantization (#2803)
* [Bugfix] Repeat and tile bug fixed, relay tests added (#2804)
* Fix caffe2 relay frontend (#2733)
* Fix a bug in nnvm to relay converter. (#2756)
* Ensure loop count is a constant before trying to unroll. (#2797)
* xcode.py: Decode bytes before output #2833
* [WIN] Fix a bug in `find_llvm` when specify llvm-config (#2758)
* [DLPACK] fix flaky ctypes support (#2759)
* [Bugfix][Relay][Frontend] Fix bug in mxnet converter for `slick_like` (#2744)
* [DOCS] Fix tutorial (#2724)
* [TOPI][Relay] Fix default `out_dtype` for `conv2d_NCHWc` and Relay (#2702)
* [Relay] fix checkwellform (#2705)
* fix prelu, now can use on 2d input and add one test (#2875)
* [CODEGEN][OPENCL] Fix compile error about ternary expression. (#2821)
* Fix Placeholder issue (#2834)
* Fix makedirs() condition in contrib (#2942)
* Add missing #!/bin/bash directive (#2951)
* Bilinear resize bug fix from PR #2777 (#2857)
* Fix `bias_add` default axis (#2829)
* Remove empty ty.rs (#2958)
* fix undefined reference to dlopen, etc (#2957)
* Removed deprecated `std::unary_function` (#2962)
* Add output format to ndk build func (#2999)
* Fix java checkstyle version (#2998)
* Fix relay invariant error message (#3011)
* Fix for caffe2 nnvm frontend (#2996)
* Fix rust resnet example (#3000)
* Fix x||!x for comparisons in rewrite simplifier (#3029)
* Fix BatchMatMulRel typerelation (#3032)
* Update dmlc-core, fix default ctors of NodeEntry (#3017)
* Fix Fuse (#3035)
* Fix PostOrderVisit signature (#3048)
* Fix winograd nnpack fp16 (#3046)
* Fix some typos (#3063, #3112)
* Fix `group_conv2d` unit test (#3113)
* Fix bug in ONNX importer (#3084)
* Fixing a doc nit (#3123)
* Fix type code error for StringImm (#3050)
* Fix bug of wrongly generated `device_map` (#2990)
* use `unordered_map` instead of map in ANF (#3024)
* Fix PRelu layout in Relay (#3013)
* Minor addition to graph runtime debug (#3129)
* Fix mali conv2d performance regression (#3131)
* Fix dense autotvm template registration in ROCm (#3136)
* Fix `conv2d_transpose` (#3138)
* Fix python lint warnings (#3145)
* Some fixes for golang latest version compiler #3119 (#3182)
* Add more syncs to fix flaky test caused by `get_valid_counts` (#3151)
* Fix AlterLayout Pass (#3155)
* Fix a multithreaded bug in llvm LazyInitJIT (#3158)
* Fix a tensorflow test bug. (#3165)
* Fix concat for ARM (#3061)
* Handle vectorize for LE statement (#3137)
* Raise exception `group_conv2d_nchw` not supported (#3195)
* Quick fix of VTA FPGA Toolchain Installation documentation (#3196)
* Check file exists before removing it (#3178)
* Fix a bug of flatten in ONNX to Relay converter (#3180)
* Fix converter where initializers were not registered as nodes (#3143)
* Fix bug in cast to bool (#3207)
* Hotfix `build_module` creation (#3198)
* Fix sort changing original input data issue (#3212)
* Fix bug in vta runtime DepPop function (#3208)
* Fix resize nearest with fractional scaling (#3244)
* Fix `vta_conv2d` crash issue after change `vta_config.json` (#3213)
* Fix a memory leak in OpManager (#3263)
* PkgConfig cause crash in PYNQ board due to link library (#3257)
* Fix Error messages in tflite.py (#3320)
* Fix typos in docs and comments (#3309, #3376)
* Bugfix min/max const canonicalize rule (#3386)
* Return module from frontend for autotvm (#3401)
* Fix constant and reshape in ONNX (#3387)
* Default verilator location fix (#3324)
* Fix autodiff for conditional expression (#3453)
* Gramatical improvements to `tensor_expr_get_started` (#3330)
* Fix AutoTVM data structure bug (#3462)
* Fix MXNet RNN without providing state initialization as input (#3326)
* Fix flaky test on topk and quantize pass (#3362)
* Add VTA PYNQ `metal_test` bitstream program logic and fix compilation issue. (#3400)
* Fix VTA function Vivado Compile Error. (#3375)
* Fix VTA DRAM functionality issue. (#3278)
* Fix reshape precompute and type error in ONNX frontend (#3230)
* Fix interpreter argument conversion for tuples. (#3349)
* Fix code generation for packed functions + tuples in VM (#3287)
* Fix memory leak in Relay interpreter (#3448)
* Fix x86 depthwise conv2d `alter_op_layout` (#3264)
* Create closure object for GlobalVar (#3411)
* Fix getting global var in prelude (#3405)
* Fix rfactor bugs which related to predicate and loop partition (#3382, #3444)
* Fix the bug in AutoTVM where SimulatedAnnealingOptimizer sometimes finds useless candidate (#3413)
* Fix name conflict in PartialEval (#3402)
* Fix int bound analysis bug for modular (#3288)
* Check arg positiveness for modular rules (#3279)
* Fixes failure of `sum` and `all` on `axis=0` (#3422)
* Fix package path in tflite test (#3427)
* Fix Windows build (#3429)
* Fix `LSTMBlockCell` in Tensorflow frontend (#3410)
* TF fix where output index is ignored (#3622)
* Runtime fix for custom datatypes (#3471)
* Relay build module warnings (#3452)
* Relay partial evaluator (#3482)
* Pynq AutoTVM tracker (#3497, #3578)
* A normal form test (#3525)
* Lint issue (#3519, #3615 )
* Any shape testing (#3528)
* Android `posix_memalign` (#3532)
* Quantization `add_rewrite` and UnifyDTypeScale (#3534)
* Bound inference fix (#3526)
* Tensorflow NCHW data format (#3514)
* First order gradient (#3550)
* JS load module example (#3556)
* Build error (#3552)
* Relay VM debug statements (#3565)
* C++ lambda expr (#3570)
* Handling of tempdir if subprocess is killed (#3574)
* Remove tabs in Chisel source (#3603)
* Relay VM DataTypeObject (#3604)
* Removing prints (#3616)
* Average Pool2D Bug (#3607)
* Missing header in `cuda_device_api.cc` (#3621)
* Tensorflow frontend fix where `output_shape` is None (#3632)
* Winograd accuracy fix (#3644)
* Fix comment (#3646)
* Zero-input op fix for recursive traversals (#3623)
* Python 3.5 compatibility (#3675)
* Fix infinite recursive `device_api.ext_dev` call in VTA. (#3843)
* Fix `depth_mult` for TensorFlow frontend (#3676)
* Fix database APIs for AutoTVM (#3821)
* Fix axis of softmax in Keras (#3834)
* Fix VTA TensorLoad module (#3841)
* Fix inconsistent python/cpp API behavior for `if_then_else`, power (#3829)
* Fix code comment of operators in ONNX frontend (#3830)
* Added repo for llvm-9 to fix missing dependency issue (#3826)
* Fix typo in Relay text parser (#3785)
* Fix tvm const warnings (#3817)
* Add gfx906 bc (#3808)
* Fixed onnx test failures when run on a cpu backend (#3764)
* Fix ArgBinder assert order (#3794)
* Fix for NoneType Target for quantization (#3792)
* Fix out-of-date quantization realize (#3790)
* Fix Qnn concatenate InferType (#3779)
* Fix dense tuning (#3768)
* Fix `visit_pattern` in ExprMutator (#3769)
* Fix Chisel Scala style (#3765)
* Fix some pass docs (#3767)
* Fix mistype in rpc tutorial (#3763)
* Fix tvm.scan follow by tvm.compute segfault (#3723)
* Fix the potential index overflow in where operator (#3751)
* Revert `compile_cmd` kwarg name change (#3746)
* Update tophub (#3752)
* Fix typo in `ir_pass.h` (#3741)
* Bug fix for VME Shell (#3737)
* Fix missing apt https transport support (#3735)
* Take zero extent loops as NoOp and remove it (#3724)
* Fix mxnet converter for hybridblock and add `div_sqrt_dim` (#3701)
* Fix partial eval unit test name (#3719)
* Fix conv2d schedule code (#3648, #3717)
* Remove thread related headers (#3713)
* Fix FunctionPass (#3712)
* Export tvm::relay::OpRegistry::OpRegistry (#3711)
* Fix Metal reinterpret (#3706)
* Fix `gather_nd` in Relay (#3442)
* Fix error in partial evaluator (#3693)
* Align the naming rule for OpAttributeUnImplemented (#3695)
* Enable the sparse schedule (#3651)
* Fix typo names in Caffe2 frontend (#3685)
* Make tests multi-process friendly. (#3683)
* Fix typo in README.md (#3684)
* Fix doc rendering  (#3897)
* Add test script starter command to document (#3993)
* Add type solver unit tests for unifying quantified funcs (#3947)
* Change Vivado install instructions to version 2018.3 (#4003)
* Add a link to the defining network description of auto-tuning tutorial (#4023)
* Additional MXNet Convolution and Deconvolution tests (#4026)
* Adding support to check if an attribute is present or not without having to get the value (#3957)
* Fix parser for cast. (#3873)
* Fix operator fusion for multiple output (#3871)
* Remove extern C warpper for cuBLAS (#3877)
* Fix int32 range overflow by using int64 (#3870)
* Remove duplicate resize (#3902)
* Fix blas cmake for mac os (#3898)
* Add another MKL name alias for MKL installed through pypi (#3853)
* Numpy compatible dtype inference for `tvm.convert` and `tvm.const` (#3861)
* Remove incorrect check for LLVM in C codegen test (#3921)
* Fix exponential blowup in interpreter (#3559)
* Fix CUDA int8x4 vectorize (#3928)
* Make buffer auto broadcast independent to the order of input args (#3956)
* Fix benchmark layout in graph tuner (#3926)
* Fix Android Demo LLVM version (#3962)
* Cast filepath arguments to string (#3968)
* Fixes "common" sub crate using nightly and main (#3965)
* Changes to make tensorize work. These changes also fix the previously broken test. (#3981)
* Remove FLOP computation when calling 3rd party library (#4005)
* Use a more intuitive way to limit the #ops in a group (#4018)
* Add more `pad_mode` support for onnx converter (#4029)
* Impose a max op limit to the op fusion pass (#4002)
* Fixes issue with CPP enums (#4019)
* Int64 shape handling for outputs. (#4031)
* [PYTHON] Fix installation for generated grammar (#4223)
* [Bugfix] Fix target host for vm compiler (#4057)
* [Fix][VM] Fix VM invoke with `set_params` (#4079)
* [Fix] Fix a few bugs when dtype is fp16 (#4088)
* [Relay][Frontend][TF] Fix Size operator (#4175)
* [cmake][ANTLR] Support setting path to ANTLR jar (#4176)
* Fix infer type of kernel in dense. (#4125)
* [Relay] Fix match case in Python-side expr functor (#4037)
* Split `adaptive_pool2d_avg` into sum and div (#4186)
* [AutoTVM] Fix Split Factors when `no_tail` is off (#4044)
* Fix extent one for the `post_stmt` in loop partition (#3734)
* [TOPI] Fix bug in intel graphics auto tune (#4093)
* [ARITH] Fix lowering of `floormod(x, y) != 0` (#4127)
* [ARITH] Fix the rule `y < x && x <= y` (#4220)
* [Bugfix][TF] reset graph after getting tag of savedmodel (#4055)
* [Fix] Fix the logic of the number of nodes checking in op fusion (#4074)
* [VTA] hotfix for de10-nano driver (#4081)
* Fixing tensor not found issue in bitserial operator (#4095)
* Fix wrong `n_trial` number in autotvm tutorials' progress bar if `n_trial` is larger then config space. (#4070)
* [PATCH] Fix undefined `__floatdihf` in libtvmruntime.so on aarch64. (#4119)
* [ARITH] Fix lowering of FloorMod (#4236)
* [Relay][Frontend][Tensorflow] Fix GatherV2 (#4238)
* Fix typing.Deque import error for Python 3.5 (#4254)
* [VTA] Hotfix for padded load test in Chisel VTA (#4264)
* [Contrib] Fix error message at `callback_get_section_size()` (#4221)
* [TOPI] Fix bug in Winograd on CUDA (#4260)
* AutoTVM: Fix hang/crash issues on feature extraction (#3689)
* [TOPI][CUDA] Fix Winograd Kernel Size Support (#4276)
* [Relay][Frontend][Tensorflow] Fix type assignment for 'tf.range' operator (#4294)
* Fix incorrect call to Unicode Win32 InetPton (#4306)
* [Relay][Frontend][Keras] handle `batch_norm` op params well (#4310)
* [VTA] fix error when `memory_id` is `VTA_MEM_ID_OUT` (#4330)
* [Doc][fix] fix sphinx parsing for pass infra tutorial (#4337)
* [Codegen] remove fp16 function override for cuda (#4331)
* [TFLite] Fix Prelu unified shape error (#4326)
* [Relay][Frontend][TF] Fix transpose when axes is not a param (#4327)
* [VTA] Bug fix for padded load with large inputs (#4293)
* Fix inconsistent operator tag name (#4134)
* Fix for a specific case when loop partitioning with indivisble. (#4243)
* Send list as argument to `schedule_conv2d` (#4358)
* [Docker] Fix TVM folder name for installing on Android and OpenCL. (#4363)
* Fix TFLite Reshape assert (#4320)
* [Relay][Frontend][TF] Fix slice when begin or size is not Const (#4372)
* Fix compilaton of bfloat16 on Windows (#4415)

### Known Issues

* The performance of Relay VM is not good enough on GPU, due to memeory allocation overhead which will be resolved later.
* TFlite rounding vs tvm rounding causing differences in accuracy and potentially off by 1 errors. For reference #3900
* TFlite pre-quantized network support is still a work in progress and the project would welcome further contributions.
* TSIM build requires `python` command exist on the host. See [forum discussion](https://discuss.tvm.ai/t/vta-build-failure/4790) for details.
* Tensorflow control flow has not been fully supported in the frontend converter.
* `topi.floor_div` is inconsistent with floor division semantic when result number is close to an integer.


### Depreciations
* Deprecating python2 support and following release (v0.6). (#2994, #2986)
* NNVM is deprecated and will be removed in a future version. (#4333, #4368)


## 0.5
This release features several major improvements. Some of the highlights are: Arbitrary bits quantization algorithm; High-level auto-differentiable programming IR -- Relay.

- Fully featured 8-bit network support
  - 8bit quantizer
  - Arbitrary bits quantization algorithm
  - Intel cpu support
  - ARM cpu support
- NVidia GPU 8-bit kernel
  - int8 gemm recipe
  - int8 conv2d
  - Autotvm integration
- Automated tuning and scheduling
  - AutoTVM optimizations for mobile GPUs
  - AutoTVM optimizations for CUDA
  - AutoTVM optimizations for x86
- Initial release of the differentiable programming IR, Relay
  - Generic & informative Relay error reporting #2408
  - Relay IR text format support #1781
  - Support control flows
  - A Normal Form Canonicalization #2251
  - Type system support
  - End to end compilation
     * Frontend support: Caffe2 #2507 , CoreML #2476 , Keras #2376 , MXNet #2163 , ONNX, TFLite #2365
     * Operator coverage #1799 #2051
  - FoldScaleAxis #2020
  - SimplifyInference #2033
  - CombineParallelConv2D #2089
  - InstrumentBoundCheckers pass #2079
  - Bind & FoldConstant #2100
  - Alter Op Layout #2150
  - General OpFusion #2090
- CodeGen
  - Gcc / g++ compatible C code generator for TVM #2161
  - Device type annotation for heterogeneous compilation #2361
  - Cache packed func ptr, lift alloca #2070
  - Generalize compute to tensor region #1476
- Runtime
  - Relay interpreter and compiler #1954
  - Heterogeneous runtime #1695
  - Language bindings: Golang runtime #1470 , Rust runtime #1597
  - Add min_repeat_ms to time_evaluator #2200
  - Bundled interpreter demonstration #2297
  - Enable PlanMemory in the graph runtime #2120
- Language Binding
  - Rust frontend #2292
- VTA
  - Improved RPC for VTA #2043
- Hybrid python programming model
  - Support for scheduling #2416
  - Support for Inter-function call  #2287
  - Backend support  #2477
- TOPI
  - Initial support for sparse tensor computation
  - Improve ARM CPU depthwise convolution performance #2345
  - Port winograd ops to relay #2356
  - Add faster-rcnn proposal op #2420
- Tutorials and docs
  - Relay language docs #2232
  - Tutorials on how to use SGX backend
  - How to write a pass in python
  - General lowering flow of TVM
  - How to do tensorize
  - TFLite frontend tutorial #2508
  - Keras seq2seq model for translation tutorial #1815
  - Committer guide and tips #2468
  - Code review guideline on API designs #2459



## 0.4

This release features several major improvements. The high-level graph optimizer is now part of TVM repo. Some of the highlights are: Initial support of AutoTVM for automated optimization; customized accelerator backend VTA.

- Tensor operator primitives
  - Introduce attrs field to operator primitives(e.g. compute) to store additional metadata, the attrs can be used as hint for scheduling
- Enable embedding of asm micro-kernels
- Hybrid python programming model
   - python AST based IR builder interface
   - support GPU programs
- AutoTVM, Automated tuning, and scheduling
   - basic autotvm infra
    - GPU IR verifier
   - basic autotuning tutorial
   - topi integration
- ARM support
    - winograd support
   - initial support of ARM autotuning records
- TOPI Vision
   - Generic GPU sort support(useful for vision)
   - SSD operator support
- TOPI numpy consistency
   - Rename all binary operators for numpy consistecy: broadcast_add-> add, broadcast_sub -> substract, broadcast_mul -> multiply, broadcast_div->divide
   - New operators: slice, LRN, equal, not_equal, less, greater
   - tutorials on topi
- Initial low-bit operator support support
    - Optimized popcount generation on ARM
    - general bit-serial convolution and GEMM
    - optimized low bit kernels
    - parallel optimization
- New topi backend optimization for intel graphics
- Adapt AVX schedules for SSE target
- VTA: customized accelerator backend
  - custom hardware backend example
  - tutorials on how to use customized accelerator
- Initial experimental support for  HLS backend
- Bugfix in SPIRV code generator for vulkan
- libdevice support, enable NVPTX backend
- Introduce NDArrayContainer for managed NDarray
- RPC and Device API
   - Support communication between big/small endian machines.
   - RPC and device API protocol upgrade (this is a non-backward compatible change) to support big-small endian communication. This is a non-backward compatible change, need to use the latest version of TVM runtime with the RPC
   - graduate rpc from contrib, tvm.contrib.rpc->tvm.rpc
   -Support tracker in Android RPC, add fault tolerance for AutoTVM
- BIG.LITTLE aware threadpool
- tvm4j graph runtime that runs end to end workload in java
- DLPack support
   - Support from_dlpack and to_dlpack
   - Enables bridges to pytorch
- Enable link of stackvm in runtime
- Tensorflow graphdef frontend
- Keras frontend
   - improved to support reuse layers, add activations
- ONNX
   - gather,  LRN
- CoreML frontend
   - Support C-RNN and activation functions
- Fix grads for sum and expand_like
- Enhanced operator fusion for multiple elemwise branches
- Separate nnvm fusion and compilation pass
- Unified build system to cmake, customizable cmake path for vulkan, rocm, cuda


## 0.3

This release features numerous improvements in TOPI and backends. We make the first step toward object detection support in TOPI, featuring operators necessary for YOLO and SSDs. The topi now supports numpy-style API and operator overloading. RPC is significantly improved to support resource allocation and using a pool of devices. We are adding two new backends: WebGL for running GPUs on the browser, and Vulkan for running on next-generation graphics API.

- TOPI Vision operators
   - SSD support
   - YOLO support
   - NMS operator support in vision
- TOPI general numpy-style operators
   - numpy style operator overload in topi
   - more operators: flip, take
   - dilation support on conv2d and depthwise
- 8bit support
    - ARM 8bit gemm
    - ARM 8bit conv
- Low bit operator support
    - popcount intrinsics
    - 1-bit fully connected
- Contrib: MPSDNN fully-connected and conv2d support
- Better RPC support
   - RPC Tracker support to allow centralized resource management
   - RPC protocol upgrade (this is a non-backward compatible change) to support timeout in the proxy
     - This is a breaking change, need to use the latest version of TVM runtime with the RPC
   - Fault-tolerant to early server termination with correct exception propagated
   - RPC support enabled for ROCm AMDGPUs
- Tutorials and docs
  - How to deploy to android devices.
- Optimizations for hardware backends
  - intel CPU (AVX and AVX512)
- Schedule Primitives
   - rfactor now support factor_axis to specify the factored dimension in the result
   - cache_write now support multiple output operators
   - enable warp memory which generates shuffle instructions
- Framework bridge
  - MXNet bridge supported
- C++ compiler API support
   - build migration
   - topi migration to c++
   - Target system in c++
- WebGL backend
   - runtime and codegen
   - topi integration
   - end to end pipeline on the browser
- Vulkan backend
   - vulkan runtime
   - spirv code generator
- Security
    - intel SGX runtime support
    - multi-threaded SGX runtime
- LLVM 7.0 support
- Robustness
   - VerifyMemory to verify incorrect GPU schedules that writes into GPU memory from cpu
   - Verify compute formulas
- Better CPU parallel runtime

## 0.2

This release comes with a complete set of TOPI support for NNVM compiler, which allows compilation of end to end workloads.
We also make major improvements in supporting new backends: ROCm for AMDGPUs and ARM GPU.

- Backend support
   - Support LLVM mainline(4.0, 5.0, 6.0)
   - Support ROCM stack for AMD GPUs
   - More robust OpenCL support for ARM GPUs
- Android RPC runtime
- Multi-threading optimization for ARM
   - multi-threaded depthwise
   - multi-threaded conv2d
- New schedule primitives
   - storage_align for shared memory alignment
   - double_buffer
- UnrollLoop : more robust version of unroll loop, count maximum steps that can be unrolled.
- Full set of TOPI operators
   - Introduce tvm.target to specify target options for compilation better.
   - broadcast/ reduction operators
   - pooling and global pooling
   - Generic target support for topi
   - schedule with external libraries
- End to end deep learning pipelines for CPU, GPU, ARM GPU
- Tutorials
  - How to load compiled module in any language runtime
  -  How to use java runtime
- Contrib library: MIOpen, CuDNN
- Ongoing items that contains functioning pieces
  - WebGL backend
  - C++ compiler support
  - MPS DNN
  - low bit support, introduced popcount


## 0.1

- Language runtime
    - python
    - javascript
    - java
    - c++
- Backend
    - arm, x86
    - javascript, wasm
    - CUDA
    - opencl
    - Metal
- DNN Library integration
- RPC  runtime
- TOPI operator pipeline python
- TOPI operator pipeline in C++
- Rough perf of the TOPI GPU pipeline
- Rough pref of TOPI CPU pipeline
- End to end graph executors


## Initial version

- Pack libary into shared library.
- External function and contrib libraries
- DLPack integration support
- AOT and module system
- Basic code structure ready.


