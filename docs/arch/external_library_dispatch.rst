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

.. _external-library-dispatch:

External Library Dispatch (BYOC)
================================

When deploying models, certain operator patterns (e.g., matmul + bias + relu) can be executed
more efficiently by vendor-optimized libraries such as cuBLAS, CUTLASS, cuDNN, or DNNL. TVM's
**BYOC (Bring Your Own Codegen)** mechanism identifies these patterns in a Relax module and
offloads them to external backends, while keeping the rest of the computation on TVM's own
generated kernels.

This document explains the BYOC pipeline: how patterns are registered, how subgraphs are
matched and extracted, how backend code generators are invoked, and how the externally compiled
code is executed at runtime.


Overview
--------

The BYOC pipeline consists of four stages:

.. code-block:: text

   IRModule (high-level Relax IR)
        │
        ▼  FuseOpsByPattern              ← match high-level ops, create composite functions
   IRModule (with Composite + Codegen attributes)
        │
        ▼  RunCodegen                    ← invoke backend codegen via FFI
   IRModule (with call_dps_packed to ExternFunc)
   + external runtime Modules
        │
        ▼  LegalizeOps + FuseOps + ...   ← compile remaining ops normally
        │
        ▼  VM compilation                ← link external modules into executable
   Deployable artifact

Each stage is a Relax transformation pass that operates on the ``IRModule``:

1. **FuseOpsByPattern** — matches operator subgraphs against registered patterns and groups them
   into composite functions annotated with ``Composite`` and ``Codegen`` attributes.
2. **MergeCompositeFunctions** (optional) — merges multiple composite functions targeting the same
   backend when inter-operator dependencies allow.
3. **RunCodegen** — finds all functions with a ``Codegen`` attribute, invokes the corresponding
   backend code generator via FFI, and replaces the original calls with ``call_dps_packed``
   to externally compiled functions.
4. **Linking** — the resulting external ``runtime.Module``\ s are attached to the ``IRModule``
   as the ``external_mods`` attribute and bundled into the final executable during
   ``relax.build()``.


Pattern Registration
--------------------

Each backend registers the operator patterns it supports in a **global pattern registry**
(``python/tvm/relax/backend/pattern_registry.py``). The registry is a static table that maps
pattern names to ``FusionPattern`` objects.

Registering patterns
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tvm.relax.backend.pattern_registry import register_patterns
   from tvm.relax.backend.patterns import make_matmul_pattern

   register_patterns([
       (
           "cublas.matmul",              # pattern name (prefix = backend)
           *make_matmul_pattern(          # returns (DFPattern, annotation_patterns)
               with_bias=False,
           ),
           _check_matmul,                # check function
       ),
       (
           "cublas.matmul_bias_relu",
           *make_matmul_pattern(
               with_bias=True,
               activation="relax.nn.relu",
           ),
           _check_matmul,
       ),
       # ... more patterns
   ])

Each entry is a tuple of ``(name, pattern, annotation_patterns, check_func)`` that gets
converted to a ``FusionPattern`` object. The name prefix (e.g., ``"cublas"``) identifies the
backend; ``get_patterns_with_prefix("cublas")`` retrieves all patterns for that backend.

Patterns registered later have **higher priority** — when a subgraph matches multiple patterns,
the highest-priority match wins.

Pattern templates
~~~~~~~~~~~~~~~~~

``python/tvm/relax/backend/patterns.py`` provides reusable templates for common patterns:

- ``make_matmul_pattern(with_bias, activation, transposed_rhs)`` — matmul with optional bias
  and activation fusion
- ``make_conv2d_pattern(with_bias, activation)`` — 2D convolution
- ``make_attention_pattern()`` — multi-head attention
- ``make_residual_block_pattern()`` — residual connections
- ``make_layer_norm_pattern()`` / ``make_rms_norm_pattern()`` — normalization layers

Each template returns ``(DFPattern, Mapping[str, DFPattern])`` — the main pattern and its
annotation sub-patterns.

Check functions
~~~~~~~~~~~~~~~

The check function validates whether a matched subgraph can actually be handled by the backend.
It receives a ``PatternCheckContext`` and returns ``True`` to accept or ``False`` to reject.

Typical checks include:

- **Data type support**: verify the operand dtypes are supported (e.g., cuBLAS supports
  float16, float32, int8, bfloat16, float8 for matmul).
- **Shape constraints**: verify reduction axes are constant, batch dimensions are compatible.
- **Leaking intermediates**: reject if an intermediate result is used outside the fused group
  (via ``has_leaking_intermediate_variables()``).


Partitioning
------------

After patterns are registered, a backend provides a **partition function** that applies
``FuseOpsByPattern`` to an ``IRModule``:

.. code-block:: python

   # python/tvm/relax/backend/cuda/cublas.py
   def partition_for_cublas(mod, bind_constants=False):
       patterns = get_patterns_with_prefix("cublas")
       return transform.FuseOpsByPattern(
           patterns, bind_constants=bind_constants, annotate_codegen=True
       )(mod)

With ``annotate_codegen=True``, each matched subgraph is wrapped in a two-level function
structure:

.. code-block:: text

   # Outer function — tagged for the codegen backend
   @R.function
   def fused_relax_matmul_cublas0(args...):
       R.func_attr({"Codegen": "cublas", "global_symbol": "fused_relax_matmul_cublas0"})
       ...
           # Inner function — identifies the specific pattern
           @R.function(private=True)
           def composite(args...):
               R.func_attr({"Composite": "cublas.matmul_bias_relu"})
               lv0 = R.matmul(x, w)
               lv1 = R.add(lv0, bias)
               lv2 = R.nn.relu(lv1)
               return lv2
       ...

The outer function carries the ``Codegen`` attribute that ``RunCodegen`` uses to dispatch to the
right backend. The inner function carries the ``Composite`` attribute that the backend codegen
uses to identify which operation to emit.

MergeCompositeFunctions
~~~~~~~~~~~~~~~~~~~~~~~

When ``annotate_codegen=False``, ``FuseOpsByPattern`` only creates inner functions with
``Composite`` attributes. A separate ``MergeCompositeFunctions`` pass then groups multiple
composite functions targeting the same backend into a single outer function with ``Codegen``
and ``global_symbol`` attributes.

This is useful when multiple sequential operations should be sent to the same backend as a
single unit (e.g., a sequence of cuBLAS matmuls that share intermediate results). The pass
checks that merging does not create cyclic dependencies between groups.


Code Generation
---------------

``RunCodegen`` (``src/relax/transform/run_codegen.cc``) is the pass that triggers backend
code generation:

1. Scan the module for all functions with a ``Codegen`` attribute.
2. Group them by backend target name.
3. For each backend, look up the registered codegen function via FFI key
   ``"relax.ext.<backend>"`` (e.g., ``"relax.ext.cublas"``).
4. Call the codegen function, which returns an array of compiled ``runtime.Module``\ s.
5. Replace the original function calls with ``call_dps_packed(ExternFunc(...), args)``.
6. Attach the compiled modules to the ``IRModule`` as the ``external_mods`` attribute.

Codegen registration
~~~~~~~~~~~~~~~~~~~~

Each backend registers a codegen function via TVM's FFI mechanism:

.. code-block:: cpp

   // src/relax/backend/contrib/cublas/codegen.cc
   ffi::Array<ffi::Module> CublasCompiler(
       ffi::Array<Function> functions,
       ffi::Map<ffi::String, ffi::Any> options,
       ffi::Map<Constant, ffi::String> constant_names) {
     ffi::Array<ffi::Module> compiled_functions;
     for (const auto& func : functions) {
       CublasJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
       serializer.serialize(func);
       auto graph_json = serializer.GetJSON();
       auto names = serializer.GetConstantNames();
       const auto pf = ffi::Function::GetGlobalRequired("runtime.CublasJSONRuntimeCreate");
       compiled_functions.push_back(
           pf(GetExtSymbol(func), graph_json, names).cast<ffi::Module>());
     }
     return compiled_functions;
   }

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = tvm::ffi::reflection;
     refl::GlobalDef().def("relax.ext.cublas", CublasCompiler);
   }

The codegen function receives:

- ``functions``: the Relax functions with ``Codegen`` attribute to compile.
- ``options``: backend-specific compilation options.
- ``constant_names``: mapping from constant values to their names (for weight handling).

It returns an array of ``runtime.Module`` objects — one per function — that contain the
externally compiled code.

Codegen strategies
~~~~~~~~~~~~~~~~~~

TVM provides two base classes for implementing backend codegens:

- **JSONSerializer** (``src/relax/backend/contrib/codegen_json/codegen_json.h``): serializes the
  composite function into a JSON graph representation. At runtime, a backend-specific JSON
  runtime module interprets the graph and dispatches to library calls. Used by cuBLAS, cuDNN,
  and most backends.

- **CSourceCodegen** (``src/relax/backend/contrib/codegen_c/codegen_c.h``): generates C/CUDA
  source code that is compiled and linked. Used when the backend requires ahead-of-time
  compilation.


Runtime Execution
-----------------

After ``RunCodegen``, the original high-level function calls are replaced with:

.. code-block:: python

   R.call_dps_packed(ExternFunc("fused_relax_matmul_cublas0"), (x, w, bias), ...)

At runtime, ``call_dps_packed`` invokes the externally compiled function through the
``PackedFunc`` interface. The external ``runtime.Module``\ s (produced by the codegen) are
imported into the final executable during ``relax.build()`` and are available via the module's
function lookup mechanism.

For JSON-based backends (cuBLAS, cuDNN), the runtime module deserializes the JSON graph and
dispatches each node to the corresponding library API call. For source-based backends, the
compiled native code is called directly.


Adding a New Backend
--------------------

To add support for a new external library:

1. **Define patterns** in ``python/tvm/relax/backend/<target>/``:

   - Create DFPatterns using templates from ``patterns.py`` or custom patterns.
   - Write check functions to validate dtypes, shapes, and other constraints.
   - Register patterns with ``register_patterns()``.
   - Provide a ``partition_for_<backend>(mod)`` convenience function.

2. **Implement codegen** in ``src/relax/backend/contrib/<target>/``:

   - Subclass ``JSONSerializer`` or ``CSourceCodegen``.
   - Implement the visitor that converts composite functions to the target format.
   - Register the codegen function as ``"relax.ext.<target>"``.

3. **Implement runtime** (for JSON-based backends):

   - Create a JSON runtime module that interprets the serialized graph and dispatches
     to the library's API calls.
   - Register the runtime constructor as ``"runtime.<Target>JSONRuntimeCreate"``.


Supported Backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Backend
     - Patterns
     - Operations
   * - cuBLAS
     - ``cublas.*``
     - Matmul (with bias, activation, transpose, dequantize variants)
   * - CUTLASS
     - ``cutlass.*``
     - Matmul, conv2d, attention, residual blocks, decode matmul
   * - cuDNN
     - ``cudnn.*``
     - Conv2d (NHWC/NCHW), stacked attention
   * - DNNL
     - ``dnnl.*``
     - Matmul, conv2d (x86 CPU). Codegen exists at C++ level; patterns are
       defined in tests rather than pre-registered.
   * - XNNPACK
     - ``xnnpack.*``
     - Static-shape ``float32`` CPU tensors for a small NHWC CNN subset:
       conv2d, optional bias, clamp-style activations, add without
       broadcasting, and no-padding 2D pooling.


XNNPACK Backend
---------------

XNNPACK support is opt-in and disabled by default. Build with
``USE_XNNPACK=ON`` to use normal CMake search paths, or with
``USE_XNNPACK=/path/to/xnnpack/prefix`` to use a specific XNNPACK install
prefix. TVM does not vendor XNNPACK and does not download it during CMake
configuration.

The current integration is a conservative CPU backend for static ``float32``
CNN/MLP islands, limited dynamic-batch ``float32`` dense/conv2d islands, and a
small stable signed-int8 QDQ subset. ``partition_for_xnnpack`` registers only
patterns that can be represented by the public XNNPACK subgraph API and leaves
all unsupported graphs on TVM's normal lowering path. Static weights and biases
must be bound into the Relax module before partitioning.

Build examples::

  cmake -S . -B build -DUSE_XNNPACK=OFF
  cmake -S . -B build -DUSE_XNNPACK=ON
  cmake -S . -B build -DUSE_XNNPACK=/path/to/xnnpack/prefix

Python usage::

  from tvm import relax
  from tvm.relax.backend.xnnpack import (
      XNNPACKCostConfig,
      XNNPACKPartitionConfig,
      XNNPACKRuntimeConfig,
      partition_for_xnnpack,
  )

  mod = relax.transform.BindParams("main", {"w": weight_np, "b": bias_np})(mod)
  config = XNNPACKPartitionConfig(
      runtime=XNNPACKRuntimeConfig(precision="fp32"),
      cost=XNNPACKCostConfig(partition_policy="greedy"),
  )
  mod = partition_for_xnnpack(mod, config=config)
  mod = relax.transform.RunCodegen({"xnnpack": config.runtime.run_codegen_options()})(mod)
  executable = tvm.compile(mod, target="llvm")
  vm = relax.VirtualMachine(executable, tvm.cpu())

Advanced partition options are passed through ``XNNPACKPartitionConfig``. The
default cost policy ``"greedy"`` partitions every supported pattern.
``"cost"`` applies a conservative heuristic before creating XNNPACK regions, so
small unary or binary islands may stay on TVM when external call overhead and
padded boundary copies are likely to dominate. ``"debug_all_supported"`` is
intended only for debugging supported-pattern coverage and is not
performance-oriented.

The cost model estimates operator count, FLOPs, input/output/constant bytes,
``XNN_EXTRA_BYTES`` padded copy bytes, graph boundaries, and visible dtype or
layout boundary costs. It accepts candidates with existing compute-heavy
operators such as supported ``conv2d`` fusions, or candidates whose
compute-to-copy ratio meets ``min_compute_to_copy_ratio``. It rejects isolated
elementwise operators by default unless ``allow_isolated_elementwise=True``.
The heuristic is intentionally simple and is not an optimal performance model.

Partition decisions can be inspected without changing runtime behavior::

  mod, report = partition_for_xnnpack(
      mod,
      config=XNNPACKPartitionConfig(
          cost=XNNPACKCostConfig(
              partition_policy="cost",
              report_partition_decisions=True,
          ),
      ),
  )

Each report entry includes stable fields such as ``candidate_id``,
``accepted``, ``reason``, ``op_list``, ``dtype``, ``layout``,
``estimated_flops``, ``copy_bytes``, ``padded_copy_bytes``,
``layout_transform_bytes``, ``cast_bytes``, boundary counts, and the selected
policy. Common reasons include ``accepted_compute_heavy``,
``accepted_ratio``, ``rejected_isolated_elementwise``,
``rejected_low_compute_to_copy_ratio``, ``rejected_unsupported_dtype``, and
``rejected_existing_support_check``.

The layout option is ``"auto"`` by default, which preserves the current strict
NHWC/OHWI policy. ``layout="preserve"`` never requests layout changes.
``layout="NHWC"`` is reported as the desired policy for cost decisions, but
the backend does not introduce broad layout rewrite or transpose insertion.
Explicit FP16 cast boundaries are likewise not lowered: ``allow_cast_boundary``
is accepted as a policy option for reporting, but explicit ``float16`` Relax
graphs remain unsupported and fall back to TVM.

Runtime options are passed to ``RunCodegen`` and are stored in the generated
XNNPACK runtime module:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``use_weights_cache``
     - Create an XNNPACK weights cache when the linked XNNPACK revision supports
       it. TVM finalizes the cache after runtime creation and before inference.
   * - ``use_workspace``
     - Create an XNNPACK workspace when ``xnn_create_runtime_v4`` and workspace
       APIs are available. The workspace is owned by the runtime module.
   * - ``profile``
     - Enable ``XNN_FLAG_BASIC_PROFILING`` when profiling APIs are available.
       The runtime module exposes ``get_profile_json`` after execution.
   * - ``dont_spin_workers``
     - Set ``XNN_FLAG_DONT_SPIN_WORKERS`` when the flag is available.
   * - ``transient_indirection_buffer``
     - Set ``XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER`` when the flag is available.
   * - ``num_threads``
     - ``1`` keeps the default caller-thread behavior. Values greater than
       ``1`` create a private pthreadpool when pthreadpool support is available.
   * - ``precision``
     - ``fp32`` keeps the default behavior. ``fp16_hint`` sets
       ``XNN_FLAG_HINT_FP16_INFERENCE`` when available. ``fp16_force`` sets
       ``XNN_FLAG_FORCE_FP16_INFERENCE`` and fails runtime creation if XNNPACK
       cannot create an FP16 runtime.

``fp16_hint`` and ``fp16_force`` are XNNPACK runtime policies only. They do not
rewrite Relax IR dtypes, do not allow explicit ``float16`` Relax graphs to be
partitioned, and do not change TVM's visible input/output dtypes. Explicit
``xnn_datatype_fp16`` lowering, mixed dtype partitioning, and FP32 static
weights or biases in FP16 partitions are left for future work.

Quantization metadata plumbing is present for the retained static signed-int8
operators. The canonical imported representation is Relax QDQ:
``relax.dequantize`` around signed-int8 tensors, a supported float Relax
operator, and a final ``relax.quantize`` back to signed int8. The runtime metadata schema
contains ``dtype``, ``qscheme`` (``none``, ``per_tensor``, or
``per_channel``), ``scale``, ``zero_point``, ``axis``, ``channel_dim``, and
``signedness``.

Supported metadata forms are scalar per-tensor parameters for ``int8``,
``uint8``, and ``int32``, and per-channel scale arrays for ``int8`` and
``int32`` weights. Scales must be static, finite, and positive; zero points
must be static and in range for the dtype; and per-channel scale length must
match the selected channel dimension. Dynamic quantization parameters,
per-channel zero-point arrays, mixed signedness, unsupported dtypes, and axis
remapping after quantized layout conversion are rejected. Runtime-owned
quantization parameter arrays are padded with ``XNN_EXTRA_QUANTIZATION_PARAMS``
where XNNPACK may overread, and their lifetime is tied to the XNNPACK runtime
or subgraph that uses them.

The TFLite Relax frontend may preserve signed-int8 quantization metadata as
QDQ graphs, but this backend currently offloads only small QDQ islands:
reshape/flatten/copy, max pooling, and same-shape residual add when their
qparams meet the backend checks. QS8 fully-connected, QS8 conv2d, QS8
depthwise conv2d, QS8 average pooling, QU8/``uint8``, dynamic-range
quantization, weight-only quantization, dynamic quantization parameters, and
unsupported quantized TFLite operators are rejected rather than silently
lowered.

Limited dynamic batch support is available as an opt-in policy:

.. code-block:: python

   mod = partition_for_xnnpack(
       mod,
       config=XNNPACKPartitionConfig(
           dynamic_shape_policy="batch_only",
           dynamic_batch_bounds={"n": 8},
       ),
   )

The default remains ``dynamic_shape_policy="none"``, which preserves the
static-shape-only checks. With ``"batch_only"``, only the leading dimension may
be symbolic. Rank, all non-batch dimensions, weights, bias, qparams, and
operator attributes must stay static. Bounds may be supplied as
``{"n": upper}``, which implies lower bound 1, or ``{"n": (lower, upper)}``.
When explicit bounds are omitted, the partitioner can read
``tir_var_upper_bound`` and optional ``tir_var_lower_bound`` function attrs.
API-provided bounds take precedence and are attached to generated XNNPACK
external functions.

Dynamic batch is supported only for ``float32`` fully-connected
(``relax.matmul`` with static rank-2 weights) and ``float32`` NHWC/OHWI
``conv2d`` with ``groups=1``. Static QS8, dynamic-range quantization,
depthwise convolution, pooling, elementwise operators, concat, resize, dynamic
H/W, dynamic channels, dynamic rank, and dynamic qparams remain unsupported.
The runtime requires public XNNPACK reshape/setup APIs:
``xnn_reshape_external_value``, ``xnn_reshape_runtime``,
``xnn_setup_runtime_v2``, and ``xnn_get_external_value_shape``. If those APIs
are not available, requesting dynamic batch fails clearly and enabled-runtime
tests skip.

At execution time the XNNPACK runtime validates actual DLTensor ranks, static
dimensions, and batch bounds. It tracks the last shape signature and reshapes
external values plus the XNNPACK runtime only when the batch size changes.
The runtime module exposes ``get_runtime_counters`` with ``reshape_count``,
``setup_count``, ``invoke_count``, and ``last_batch_size`` for debugging. Size
calculations for element counts, byte counts, padded buffers, and quantization
parameter padding use checked multiplication and fail before allocation on
overflow-like shapes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Relax pattern
     - Restrictions
   * - ``relax.nn.conv2d``
     - NHWC input/output, OHWI static weights, ``groups=1``, static bias only
       when fused through ``relax.add``.
   * - ``relax.nn.relu`` and ``relax.clip``
     - Static ``float32`` tensors. ReLU and ReLU6 are represented as XNNPACK
       clamp nodes.
   * - ``relax.sigmoid`` and ``relax.tanh``
     - Static ``float32`` tensors.
   * - ``relax.nn.gelu`` and ``relax.nn.gelu_tanh``
     - Static ``float32`` tensors. ``gelu_tanh`` maps to XNNPACK's approximate
       GELU unary op. Isolated GELU islands can be rejected by the cost policy.
   * - ``relax.matmul`` + static bias + GELU
     - Static rank-2 float32 input, static rank-2 float32 weights in
       ``[input_channels, output_channels]`` form, static float32 bias, and
       either exact GELU or approximate GELU. This is intended for small MLP
       blocks, not batch matrix multiply or full attention lowering.
   * - ``relax.nn.softmax``
     - Static float32 tensors, last axis only. Non-last-axis softmax and
       ``relax.nn.log_softmax`` are intentionally rejected.
   * - ``relax.add``
     - Equal static input shapes only. Broadcasting is intentionally rejected.
   * - ``relax.nn.max_pool2d`` and ``relax.nn.avg_pool2d``
     - NHWC input/output, dilation 1, ``ceil_mode=False``, and zero padding.
   * - QDQ ``relax.reshape`` / ``relax.flatten`` / copy
     - Static signed-int8 tensors with exactly matching input/output scale and
       zero point. The copy case is represented as
       ``dequantize(int8) -> quantize(int8)`` with unchanged shape and qparams.
   * - QDQ ``relax.nn.max_pool2d``
     - Static signed-int8 NHWC tensors, constant qparams, exactly matching
       input/output qparams, static pool/stride/padding/dilation,
       ``ceil_mode=False``.
   * - QDQ ``relax.add``
     - Static signed-int8 tensors, exactly equal input shapes, constant
       per-tensor qparams, no scalar or channel broadcasting, and optional
       ReLU/ReLU6/clip fusion.
   * - Dynamic-batch ``relax.matmul``
     - Opt-in with ``dynamic_shape_policy="batch_only"``. Float32 input/output,
       symbolic leading batch only, finite positive batch bounds, static rank-2
       weights, optional static float32 bias, and optional ReLU/ReLU6/clip.
   * - Dynamic-batch ``relax.nn.conv2d``
     - Opt-in with ``dynamic_shape_policy="batch_only"``. Float32 NHWC
       input/output, symbolic leading batch only, finite positive batch bounds,
       OHWI static weights, ``groups=1``, static attributes, optional static
       float32 bias, and optional ReLU/ReLU6/clip.

There is no full attention lowering, batch matrix multiply, SwiGLU,
``log_softmax``, int8 multiply/subtract/concat/pad/resize, generic spatial
mean, QS8 fully-connected, QS8 conv2d, QS8 depthwise conv2d, QS8 average
pooling, dynamic-range quantization, QU8/``uint8``, 4-bit, weight-only quantization,
dynamic qparams, layout conversion, dynamic-shape support, broad broadcasting,
or broad CNN coverage. Explicit ``float16`` Relax graphs are
also unsupported and must fall back to TVM. Dynamic-shape support is limited to
the explicit batch-only cases above; arbitrary symbolic shapes still fall back
to TVM. The cost policy can reject isolated small fp32 or int8 elementwise,
unary, and reshape/copy islands, even when the greedy/debug policies would
partition them. Dynamic-batch report entries set
``dynamic_batch=True`` and include the symbol name, lower/upper bounds, and
min/max FLOP and copy-byte estimates.

The runtime uses XNNPACK's public ``xnnpack.h`` API only. It initializes
XNNPACK with ``xnn_initialize`` and does not include
``xnnpack/experimental.h``. By default the runtime creates XNNPACK subgraphs
with a null threadpool so execution remains single-threaded on the caller
thread. Runtime-owned input, output, and static constant buffers are padded by
``XNN_EXTRA_BYTES``. Copied static constants, optional weights cache, optional
workspace, optional pthreadpool, subgraph, and runtime handles are owned by the
runtime module and released when the module is destroyed.

Runtime validation is deliberately strict. XNNPACK JSON modules validate graph
metadata when the runtime module is created: every node must carry shape and
dtype metadata, kernel nodes must use a supported ``op_kind`` and required
operator attrs, node references must point to existing tensor entries, and graph
outputs must be valid. Constants are checked during runtime initialization so
their dtype, rank, shape, compact layout, device, byte offset alignment, and
byte size match the serialized metadata before XNNPACK sees the pointer.

External tensors are checked on every invocation. The runtime rejects non-CPU
tensors, dtype mismatches, rank mismatches, static-dimension mismatches,
non-positive dimensions, non-compact strides, and unaligned byte offsets. For
dynamic batch, the actual leading dimension must stay within the configured
lower/upper bounds while all non-batch dimensions remain static. Size
calculations for element counts, tensor bytes, padded tensor bytes, and
quantization-parameter arrays use checked arithmetic and fail before allocation
when a shape would overflow host ``size_t``.

Quantization metadata validation checks that scales are finite and positive,
zero points are in range for the signedness and dtype, per-channel axes are
valid, scale length matches the channel dimension, signedness matches dtype,
and padded scale arrays account for ``XNN_EXTRA_QUANTIZATION_PARAMS``. Invalid
metadata fails with explicit messages such as ``scales must be finite and
positive``, ``zero_point must be in [-128, 127]``, ``scale length must match
channel_dim``, or ``axis must match channel_dim``.

Typical validation failures look like ``tensor dtype mismatch``, ``tensor rank
mismatch``, ``tensor shape mismatch at dim 0``, ``tensor must be compact``,
``dynamic batch exceeds the configured upper bound``, ``Unsupported XNNPACK
JSON op_kind``, or ``tensor byte size overflows size_t``. These failures are
intentional: invalid or malformed XNNPACK regions must fail clearly rather than
silently falling back to incorrect execution.

When available, TVM prefers ``xnn_create_runtime_v4`` so weights cache,
workspace, threadpool, and runtime flags can be configured together. If v4 is
not available, TVM falls back to v3 for weights-cache-only configurations, or
to v2 for the default runtime. Unsupported requested options fail clearly.
The current layout policy is strict: supported convolutions use NHWC input and
output tensors with OHWI weights, and the partitioner does not insert layout
transposes. Runtime tensors must be compact CPU tensors.

Unsupported operators and unsupported attributes are not partitioned. They
continue through TVM's normal CPU lowering path, and mixed graphs may contain
both TVM and XNNPACK regions.

Benchmarking and validation::

  python tests/python/relax/benchmark_xnnpack.py --model xnnpack_tiny_cnn
  python tests/python/relax/benchmark_xnnpack.py --model xnnpack_tiny_cnn --partition-policy cost --report-partition-decisions
  python tests/python/relax/benchmark_xnnpack.py --model xnnpack_tiny_cnn --use-weights-cache --use-workspace --profile
  python tests/python/relax/benchmark_xnnpack.py --model xnnpack_tiny_cnn --precision fp16_hint
  python tests/python/relax/benchmark_xnnpack.py --quantization-mode static_qs8 --report-partition-decisions
  python tests/python/relax/benchmark_xnnpack.py --model torchvision:mobilenet_v2

The in-tree ``xnnpack_tiny_cnn`` benchmark uses only supported NHWC ``float32``
operators and compares normal TVM CPU execution with XNNPACK BYOC execution.
``--quantization-mode static_qs8`` uses an in-tree signed-int8 QDQ fixture with
no TensorFlow or PyTorch dependency. The benchmark prints platform and
architecture information, detected XNNPACK feature flags, partition counts,
partition-report reason summaries and byte estimates when requested, p50/p90/p99
latency, first-run latency, steady-state latency, optional memory deltas, and
XNNPACK profiling summaries when profiling is both requested and available.
The optional ``torchvision:*`` path is best-effort and may report zero XNNPACK
partitions for models that rely on unsupported depthwise convolution, dense
layers, NCHW layout, or other unsupported operators.

For future explicit FP16 experiments, run TVM mixed-precision rewrites before
partitioning and inspect the resulting dtype and cast boundaries before enabling
XNNPACK partitioning::

  mod = tvm.relax.transform.ConvertToDataflow()(mod)
  mod = tvm.relax.transform.ToMixedPrecision(out_dtype="float32")(mod)
  # Future work: partition_for_xnnpack(mod, precision="explicit_fp16")

Runtime precision hints may change XNNPACK's internal compute path and accuracy.
Benchmark output should be treated as measured data for the local hardware only;
TVM does not fabricate speedup results.

Troubleshooting:

* If ``xnnpack_enabled`` is false in the benchmark output, rebuild TVM with
  ``USE_XNNPACK=ON`` or ``USE_XNNPACK=/path/to/xnnpack/prefix``.
* If the partition count is zero, inspect the model for unsupported dtype,
  symbolic shapes, NCHW layout, dynamic weights, broadcasting, or unsupported
  operators.
* If numerical validation fails, confirm the input tensors are compact CPU
  tensors and that static parameters were bound before partitioning.
* If a runtime option fails, inspect
  ``runtime.XNNPACKJSONRuntimeGetCapabilities`` or the benchmark's
  ``xnnpack_capabilities`` output to confirm the linked XNNPACK revision
  exposes the required public APIs.
* If CMake fails during feature probing, verify that the configured
  ``xnnpack.h`` and XNNPACK library come from the same external installation.
  TVM fails configure only for baseline public APIs required by the current
  runtime; optional FP16, QS8, workspace, and profiling
  features are reported as unavailable instead.
* Dynamic-range, weight-only, and QU8 quantization are not part of the cleaned
  backend. Use normal TVM lowering for those models until a separate tested
  implementation is added.

Deployment and platform notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XNNPACK remains an external dependency. TVM does not vendor XNNPACK, does not
download it from CMake, and does not add it to the default build. The
recommended deployment flow is:

1. Build and install XNNPACK for the target platform with the platform's normal
   CMake toolchain.
2. Configure TVM with ``USE_XNNPACK=/path/to/xnnpack/prefix`` using the same
   compiler and ABI.
3. Run the XNNPACK Relax smoke tests and the benchmark script on the target, or
   through the platform's normal remote execution flow.

This integration has local smoke coverage only for the developer machine used
to build the patch. The following platform commands are maintainer reproduction
recipes, not claims that every platform was tested as part of this change.

Linux x86_64 and Linux aarch64::

  cmake -S /path/to/XNNPACK -B /tmp/xnnpack-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/xnnpack
  cmake --build /tmp/xnnpack-build --target install -j

  cmake -S /path/to/tvm -B /tmp/tvm-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_XNNPACK=/opt/xnnpack
  cmake --build /tmp/tvm-build --target tvm_runtime tvm_compiler -j

  python tests/python/relax/test_codegen_xnnpack.py -q
  python tests/python/relax/benchmark_xnnpack.py --model xnnpack_tiny_cnn --number 10 --repeat 3
  python tests/python/relax/benchmark_xnnpack.py --quantization-mode static_qs8 --number 10 --repeat 3

For Linux shared builds, ensure the XNNPACK, pthreadpool, cpuinfo, and
microkernel libraries are discoverable by the runtime loader. For static builds,
link all dependent XNNPACK libraries into the TVM runtime binary or final
application. FP16 availability depends on the target CPU and XNNPACK runtime
creation flags; ``fp16_force`` may fail clearly on hardware that cannot honor
the request. QS8 paths require the signed-int8 datatype and subgraph APIs
reported by ``runtime.XNNPACKJSONRuntimeGetCapabilities``.

Android arm64-v8a::

  cmake -S /path/to/XNNPACK -B /tmp/xnnpack-android \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-23 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/xnnpack-android
  cmake --build /tmp/xnnpack-android --target install -j

  cmake -S /path/to/tvm -B /tmp/tvm-android \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-23 \
        -DUSE_XNNPACK=/opt/xnnpack-android

Use the same NDK, ABI, API level, and C++ runtime for XNNPACK and TVM. Run smoke
tests through the existing TVM Android RPC or app deployment flow. Multi-thread
configuration requires pthreadpool support in the linked XNNPACK build; the
default ``num_threads=1`` path keeps caller-thread execution.

iOS arm64::

  cmake -S /path/to/XNNPACK -B /tmp/xnnpack-ios \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_SYSROOT=iphoneos \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/xnnpack-ios
  cmake --build /tmp/xnnpack-ios --target install -j

  cmake -S /path/to/tvm -B /tmp/tvm-ios \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_SYSROOT=iphoneos \
        -DUSE_XNNPACK=/opt/xnnpack-ios

iOS deployments usually prefer static linking into the final application. Keep
bitcode, minimum deployment target, C++ standard library, and symbol visibility
settings consistent between XNNPACK, TVM, and the host app. Run validation in an
iOS simulator or on-device test harness; these platform tests are not part of
default TVM CI.

Emscripten wasm32 with SIMD::

  emcmake cmake -S /path/to/XNNPACK -B /tmp/xnnpack-wasm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-msimd128" \
        -DCMAKE_CXX_FLAGS="-msimd128" \
        -DCMAKE_INSTALL_PREFIX=/opt/xnnpack-wasm
  cmake --build /tmp/xnnpack-wasm --target install -j

  emcmake cmake -S /path/to/tvm -B /tmp/tvm-wasm \
        -DUSE_XNNPACK=/opt/xnnpack-wasm

Emscripten pthreads, SIMD, and memory settings must match between XNNPACK, TVM,
and the final web application. Use ``num_threads=1`` unless the web deployment
has SharedArrayBuffer and pthreads configured. WASM benchmark results are highly
browser- and flag-dependent; record browser, engine, SIMD, pthread, and memory
settings with every result.

Optional maintainer CI recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default TVM CI should remain unchanged and must not require XNNPACK. A
maintainer-run XNNPACK Linux job can be reproduced with:

1. Install XNNPACK externally into a known prefix.
2. Configure TVM with ``USE_XNNPACK=/path/to/prefix``.
3. Build ``tvm_runtime`` and ``tvm_compiler``.
4. Run ``pytest tests/python/relax/test_codegen_xnnpack.py -q``.
5. Run a benchmark dry-run, for example
   ``python tests/python/relax/benchmark_xnnpack.py --number 1 --repeat 1`` and
   ``python tests/python/relax/benchmark_xnnpack.py --quantization-mode static_qs8 --number 1 --repeat 1``.

Android, iOS, and WASM jobs should remain manual until the project agrees on an
external-dependency CI policy for XNNPACK.


Source Code Map
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Path
     - Contents
   * - ``python/tvm/relax/backend/pattern_registry.py``
     - Pattern registry API (register_patterns, get_patterns_with_prefix)
   * - ``python/tvm/relax/backend/patterns.py``
     - Reusable pattern templates (make_matmul_pattern, etc.)
   * - ``python/tvm/relax/backend/cuda/cublas.py``
     - cuBLAS patterns and partition_for_cublas
   * - ``python/tvm/relax/backend/cuda/cutlass.py``
     - CUTLASS patterns and partition_for_cutlass
   * - ``python/tvm/relax/backend/cuda/cudnn.py``
     - cuDNN patterns and partition_for_cudnn
   * - ``python/tvm/relax/backend/xnnpack.py``
     - XNNPACK pattern registration and partition helper
   * - ``src/relax/backend/pattern_registry.cc``
     - Pattern registry C++ implementation
   * - ``src/relax/transform/run_codegen.cc``
     - RunCodegen pass (CodeGenRunner)
   * - ``src/relax/transform/merge_composite_functions.cc``
     - MergeCompositeFunctions pass
   * - ``src/relax/backend/contrib/cublas/codegen.cc``
     - cuBLAS codegen (JSONSerializer-based)
   * - ``src/relax/backend/contrib/cutlass/codegen.cc``
     - CUTLASS codegen
   * - ``src/relax/backend/contrib/codegen_json/codegen_json.h``
     - JSONSerializer base class
   * - ``src/relax/backend/contrib/codegen_c/codegen_c.h``
     - CSourceCodegen base class
