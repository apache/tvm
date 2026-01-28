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

# Example NPU Backend

A hands-on example showing how to build a Neural Processing Unit (NPU) backend for TVM's Relax framework using Bring Your Own Codegen (BYOC).

## What This Is

This is an educational template that demonstrates real NPU concepts without requiring actual NPU hardware. It shows developers how to:

- **Pattern-based partitioning**: Identify and group operations that should run on specialized hardware
- **Memory hierarchy management**: Handle different memory tiers (L0/L1/L2/L3) common in NPUs
- **Automatic tiling**: Break large tensors into smaller chunks that fit in on-chip memory
- **Quantization support**: Handle different data precisions efficiently
- **BYOC integration**: Connect custom backends to TVM's compilation pipeline
- **Operator availability checking**: Gracefully handle operators that may not be available in all TVM builds

## Quick Start

```python
import tvm
from tvm import relax
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, RunCodegen

# Import to register patterns
import tvm.relax.backend.contrib.example_npu

# Get available patterns
patterns = get_patterns_with_prefix("example_npu")
print(f"Available patterns: {[p.name for p in patterns]}")

# Your model gets automatically partitioned
# Operations matching patterns get fused into "Composite" functions
# Those get lowered to the example NPU backend
```

The snippet above shows how to discover registered patterns. A minimal runnable example that demonstrates the BYOC flow (partition -> merge -> codegen) using the example test module looks like this:

```python
# This imports the example module used in the tests. Importing the test
# module path directly works when running from the repo root (pytest does
# this automatically).
from tests.python.contrib.test_example_npu import MatmulReLU
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions, RunCodegen
import tvm.relax.backend.contrib.example_npu  # registers patterns

mod = MatmulReLU
patterns = get_patterns_with_prefix("example_npu")

# Apply partitioning and codegen annotation
mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod = MergeCompositeFunctions()(mod)
mod = RunCodegen()(mod)

print(mod)
```

A compact visualization of the BYOC flow:

```
Model source (Relax)
	│
	▼
Pattern-based partition (FuseOpsByPattern)
	│
	▼
Composite functions (MergeCompositeFunctions)
	│
	▼
Lower/Codegen for example NPU (RunCodegen / relax.ext.example_npu)
	│
	▼
Runtime dispatch to NPU runtime (runtime.ExampleNPUJSONRuntimeCreate)
```

## Supported Operations

The backend recognizes these common neural network patterns:

### Core Operations (always available)
- `example_npu.dense` - Dense/fully connected layers
- `example_npu.matmul` - Matrix multiplication operations
- `example_npu.conv1d` - 1D convolution for sequence processing
- `example_npu.conv2d` - 2D convolution for image processing
- `example_npu.depthwise_conv2d` - Depthwise separable convolutions
- `example_npu.max_pool2d` - 2D max pooling
- `example_npu.avg_pool2d` - 2D average pooling
- `example_npu.batch_norm` - Batch normalization

### Activation Functions (availability depends on TVM build)
- `example_npu.relu` - ReLU activation
- `example_npu.relu6` - ReLU6 activation (if available)
- `example_npu.sigmoid` - Sigmoid activation (if available)
- `example_npu.tanh` - Hyperbolic tangent (if available)
- `example_npu.gelu` - Gaussian Error Linear Unit (if available)

### Element-wise Operations
- `example_npu.add` - Element-wise addition
- `example_npu.multiply` - Element-wise multiplication
- `example_npu.subtract` - Element-wise subtraction
- `example_npu.divide` - Element-wise division

### Quantization Support
- `example_npu.quantize` - Quantization operations (if available)
- `example_npu.dequantize` - Dequantization operations (if available)

### Fused Patterns
- `example_npu.conv2d_relu_fused` - Optimized Conv2D+ReLU fusion

**Note**: Some operators may not be available in all TVM builds. The backend automatically skips registration for unavailable operators.

## Files

### Backend Implementation
- `patterns.py` - Defines which operations get fused together, along with pattern metadata and architectural annotations used by the partitioner. Includes operator availability checking and NPU-specific constraints.
- `__init__.py` - Registers the backend and its BYOC entry points with TVM so the compiler can discover and use the example NPU.

### Runtime Implementation
- `src/runtime/contrib/example_npu/example_npu_runtime.cc` - C++ runtime implementation that handles JSON-based graph execution for the NPU backend.

### Tests and Examples
- `tests/python/contrib/test_example_npu.py` - Comprehensive test suite containing example IRModules (e.g. `MatmulReLU`, `Conv2dReLU`) and demonstrating the complete BYOC flow from pattern registration to runtime execution.

## Status / Build

- The example backend is an educational, CPU-backed emulation. It does not require real NPU hardware.
- The backend includes robust operator availability checking - patterns are only registered for operators that exist in the current TVM build.
- Tests and runtime features are skipped automatically when the example codegen/runtime are not built into TVM. The test checks for the presence of these global functions before running:

```python
import tvm
has_codegen = tvm.get_global_func("relax.ext.example_npu", True)
has_runtime = tvm.get_global_func("runtime.ExampleNPUJSONRuntimeCreate", True)
has_example_npu = has_codegen and has_runtime
```

If `has_example_npu` is False, tests are skipped. This ensures compatibility across different TVM build configurations.

## Testing

Run the tests to see it in action:

```bash
pytest tests/python/contrib/test_example_npu.py -v
```

Tests are skipped if the backend isn't built — see the test file for the exact runtime/codegen checks. Running `pytest` from the repository root ensures imports like `tests.python.contrib.test_example_npu` resolve correctly.

The test suite includes:
- Pattern registration verification (checks that core patterns are available)
- Graph partitioning validation (ensures operations get grouped correctly)
- End-to-end execution testing (verifies runtime integration)
- Operator availability testing (graceful handling of missing operators)

### Example output

When you run the quick-start snippet or the test, you should see output similar to the following (truncated for brevity):

```
Available patterns: ['example_npu.dense', 'example_npu.matmul', 'example_npu.conv1d', 'example_npu.conv2d', 'example_npu.depthwise_conv2d', 'example_npu.max_pool2d', 'example_npu.avg_pool2d', 'example_npu.batch_norm', 'example_npu.relu', 'example_npu.add', 'example_npu.multiply', 'example_npu.conv2d_relu_fused']

Relax IRModule
def @main(...) -> ...
	%0 = call_extern("relax.ext.example_npu", ...)

# composite functions
def @composite_0(...) /* Composite */ = ...
```

This shows the registered patterns and that matched subgraphs were turned into composite functions and lowered to the example NPU codegen/runtime.

## Key Features Demonstrated

### NPU Architectural Concepts
- **Multi-tier memory hierarchy**: SRAM (256KB), CMX (512KB), and DRAM management
- **Tiling constraints**: 32x32 tiles with 16-element vectors for optimal NPU utilization
- **Quantization support**: INT8/INT16 for inference acceleration, mixed precision handling
- **Specialized execution units**: Matrix engines (16x16), vector units (64-wide), pooling units
- **Power management**: Support for different power modes (high_performance, balanced, low_power)

### Pattern Matching Features
- **Operator availability detection**: Gracefully handles missing operators in different TVM builds
- **Memory constraint checking**: Validates tensor sizes against NPU memory limits
- **Fusion opportunities**: Identifies conv+activation and other beneficial fusions
- **Layout preferences**: NHWC channel-last layouts preferred by NPUs

### Error Handling
- **Robust exception handling**: Uses specific `TVMError` instead of generic exceptions
- **Graceful degradation**: Continues operation when optional operators are unavailable
- **Comprehensive testing**: Validates both successful cases and error conditions

## Context

NPUs are specialized for neural network workloads and can be 10-100x more efficient than general-purpose CPUs/GPUs for inference. This example shows the architectural patterns you'll encounter when building real NPU backends, making it easier to adapt to specific hardware like:

- Mobile NPUs (AMD XDNA, Google Edge TPU, Samsung NPU)
- Dedicated AI chips (Intel Movidius, Qualcomm Hexagon, MediaTek APU)
- Cloud AI accelerators (AWS Inferentia, Google TPU,  Microsoft Azure Maia)
- Custom ASIC designs and embedded AI processors

## Learn More

This backend serves as both a working example and educational resource for understanding NPU integration patterns. The implementation demonstrates vendor-neutral concepts that apply across different NPU architectures, making it a valuable starting point for real NPU backend development.
