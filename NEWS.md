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

This file records the changes in TVM library in reverse chronological order.

## On-going version

Refer to the Roadmap issue for complete list on on-going version features.
If you check in something that is not reflected in Roadmap issue, please reply
to that issue so it can get added.

## 0.6

### Relay in Production
Relay is a functional, differentiable programming language designed to be an expressive intermediate representation for machine learning systems. Relay supports algebraic data types, closures, control flow, and recursion, allowing it to directly represent more complex models than computation graph-based IRs (e.g., NNVM) can. In TVM v0.6, Relay is in stable phase and is ready for production.

* Algebraic Data Types (ADT) support (#2442, #2575). ADT provides an expressive, efficient, and safe way to realize recursive computation (e.g., RNN). Refer to https://docs.tvm.ai/langref/relay_adt.html for more information.
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
* [RUTNIME] Support C++ RPC (#4281)
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
* Fixes "common" sub crate using nightly and master (#3965)
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
* Deprecating python2 support in the master branch and following release (v0.6). (#2994, #2986)
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


