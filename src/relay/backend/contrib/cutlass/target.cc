/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/cutlass/target.cc
 * \brief Registers the "cutlass" external codegen TargetKind.
 */

#include <tvm/target/target.h>

#include "./codegen.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cutlass {

/*!
 * \brief This external codegen target can use the CUTLASS template library included in
 * TVM's 3rdparty/cutlass.
 *  - Patterns: python/tvm/relay/op/contrib/cutlass.py
 *  - Custom compiler: python/tvm/contrib/cutlass/build.py,
 *                     src/relay/backend/contrib/cutlass/codegen.cc
 */
TVM_REGISTER_TARGET_KIND("cutlass", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true))
    .set_attr<tvm::transform::Pass>("RelayToTIR", CompileForCutlass())
    // An integer specifying the compute capability. For example, 75 for Turing and
    // 80 or 86 for Ampere.
    .add_attr_option<Integer>("sm", Integer(80))
    // Whether to use slower but very accurate (compared to tf32) 3xtf32 mode for
    // fp32 inputs on tensorcore.
    .add_attr_option<Bool>("use_3xtf32", Bool(true))
    // Split factor candidates for split-K GEMM. If split-K > 1, the GEMM K-loop is computed in
    // parallel across split-K blocks, and a separate global reduction kernel is launched to
    // accumulate partial reductions. The profiler will pick the best split-k factor from the
    // given candidate list. Note that the larger split-K factor requires a larger workspace.
    // Currently, parallel split-k has been tested only for wgrad. For GEMM and other conv2d
    // kinds, split_k_slices is ignored.
    .add_attr_option<Array<Integer>>("split_k_slices", Array<Integer>({1}))
    // When True, profile all kernel variants with smaller alignments than the largest possible.
    .add_attr_option<Bool>("profile_all_alignments", Bool(false))
    // Whether to profile all candidate kernels, or stop profiling after the first applicable kernel
    // is found.
    .add_attr_option<Bool>("find_first_valid", Bool(false))
    // Whether to compile profiler executables for different kernels in parallel.
    .add_attr_option<Bool>("use_multiprocessing", Bool(false))
    // Number of threads to use during compilation, or -1 to use number of cpus.
    .add_attr_option<Integer>("threads", Integer(-1))
    // Whether to replace sigmoid with tanh.
    .add_attr_option<Bool>("use_fast_math", Bool(false))
    // A temporary directory where intermediate compiled artifacts will be stored.
    .add_attr_option<String>("tmp_dir", String("./tmp"));

}  // namespace cutlass
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
