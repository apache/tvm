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
 * \file tvm/tir/transform.h
 * \brief TIR specific transformation passes.
 */
#ifndef TVM_TIR_TRANSFORM_H_
#define TVM_TIR_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>

#include <string>

namespace tvm {
namespace tir {
namespace transform {

using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassContextNode;
using tvm::transform::PassInfo;
using tvm::transform::PassInfoNode;
using tvm::transform::PassNode;
using tvm::transform::Sequential;

/*
 * \brief Create a function pass that optimizes PrimFuncs.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required);

/*!
 * \brief Function level pass that applies transformations to all
 *        TIR functions within the module.
 */
class PrimFuncPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The pass function called on each. */
  runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("pass_info", &pass_info); }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param pass_ctx The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "tir.PrimFuncPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncPassNode, PassNode);
};

class PrimFuncPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL PrimFuncPass(
      runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(PrimFuncPass, Pass, PrimFuncPassNode);
};

/*
 * \brief Create a function pass that optimizes PrimFuncs. Specialized to lambdas.
 *
 * \param pass_func Lambda function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
template <typename F>
Pass CreatePrimFuncPass(F pass_func, int opt_level, String name, tvm::Array<String> required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return PrimFuncPass({pass_func, name}, pass_info);
}

/*!
 * \brief Inject prefetch instructions into stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectPrefetch();

// TODO(tvm-team): consolidate configs to the PassContext
/*!
 * \brief Flatten the multi-dimensional read/write
 *  to single dimensional Load/Store
 *
 * \param cache_line_size The size of CPU cache line.
 * \param create_bound_attribute Whether to create bound attributes.
 *
 * \return The Pass
 */
TVM_DLL Pass StorageFlatten(int cache_line_size, bool create_bound_attribute = false);

/*!
 * \brief Inject copy intrinsics with optional pad.
 *
 * \param pragma_key The pragma key for hint of copy.
 * \param fintrin The function with signature
 *
 *   Stmt fintrin(Buffer src,
 *                Buffer dst,
 *                Array<Expr> pad_before,
 *                Array<Expr> pad_after,
 *                Expr pad_value)
 * \return The pass.
 */
TVM_DLL Pass InjectCopyIntrin(String pragma_key, runtime::PackedFunc fintrin);

/*!
 * \brief Detect and insert sync points to co-processor.
 *
 * \return The pass.
 */
TVM_DLL Pass CoProcSync();

/*!
 * \brief Lift common attrs with attr_key to outer scope.
 *
 * \param attr_key The attribute key to be checked.
 * \return The pass.
 */
TVM_DLL Pass LiftAttrScope(String attr_key);

/*!
 * \brief partition loops in the stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass LoopPartition();

/*!
 * \brief Lower vectorization loops.
 *
 * \param enable_vectorize Whether vectorization is enabled.
 *
 * \return The pass.
 */
TVM_DLL Pass VectorizeLoop(bool enable_vectorize = true);

/*!
 * \brief Inject virtual thread loops.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectVirtualThread();

/*!
 * \brief Inject double buffer statements.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectDoubleBuffer();

/*!
 * \brief Rewrite storage allocation pattern.
 *  Moves the allocation to outer most possible scope.
 *  Trying to share space between allocations to make
 *  a static allocation plan when possible.
 *
 * \return The pass.
 */
TVM_DLL Pass StorageRewrite();

/*!
 * \brief unroll the constant loop marked by unroll.
 * This pass also automatically attach pragma unroll tag to loops which meets the standard.
 *
 * \return The pass.
 */
TVM_DLL Pass UnrollLoop();

/*!
 * \brief Remove No Op from the Stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass RemoveNoOp();

/*!
 * \brief Detect and rewrite unsafe select that contains memory access.
 *
 * \return The pass.
 */
TVM_DLL Pass RewriteUnsafeSelect();

/*!
 * \brief Run arithmetic simplifications on the statements and expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass Simplify();

/*!
 * \brief Instruments bound checkers.
 *
 * \return The pass.
 */
TVM_DLL Pass InstrumentBoundCheckers();

/*!
 * \brief Transform the high-level PrimFunc to a low-level version
 *        that can be used as an API function.
 *
 *
 *  The main task of this function is to create code to :
 *   - Map the values in the api_args to Var that is required by body.
 *   - Insert assertions to check type/value of the passed arguments.
 *
 * \param num_unpacked_args Number of arguments that
 *         are processed in plain form instead of packed form.
 *
 * \note
 *  The function signature have two cases
 *
 *  let num_packed_args = len(api_args) - num_unpacked_args;
 *
 *  if num_packed_args is zero:
 *     f(api_arg_0, api_arg_1, .., api_arg_n) where n == len(api_args)
 *
 *  if num_packed_args is not zero:
 *       f(TVMArg* packed_args, int* packed_arg_type_ids, int num_packed_args,
 *         api_arg_k, api_arg_k+1, ... api_arg_n,
 *         TVMValue* out_ret_val, int* out_ret_tcode)
 *
 *       where n == len(api_args), k == num_packed_args
 *
 * \return The pass.
 */
TVM_DLL Pass MakePackedAPI(int num_unpacked_args);

/*!
 * \brief Remap the thread axis
 *
 *  This can be used to get equivalent program which uses
 *  threadIdx.y in place of threadIdx.x by passing
 *  {"threadIdx.x": thread_axis("threadIdx.y")}
 *
 *
 * \return The pass.
 */
TVM_DLL Pass RemapThreadAxis(Map<String, IterVar> axis_map);

/*!
 * \brief Lower custom datatypes.
 *
 * See tvm::datatypes::Registry for more information on adding custom datatypes.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerCustomDatatypes();

/*!
 * \brief Decorate all the function's body as device function.
 *
 * \return The pass.
 */
TVM_DLL Pass DecorateDeviceScope();

/*!
 * \brief Split the function into a host function and device functions.
 *
 * \return The pass.
 */
TVM_DLL Pass SplitHostDevice();

/*!
 * \brief skip assert stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass SkipAssert();

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 *
 * \param storage_scope The storage scope considered.
 * \return The pass.
 */
TVM_DLL Pass ThreadSync(String storage_scope);

/*!
 * \brief Lower cross thread alleduce.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerThreadAllreduce();

/*!
 * \brief Infer the TensorCore fragment infomation using tensor intrinsics
 *
 * \return The pass.
 */
TVM_DLL Pass InferFragment();

/*!
 * \brief Lower builtin intrinsics.
 * \return The pass.
 */
TVM_DLL Pass LowerTVMBuiltin();

/*!
 * \brief Lower the target specific function intrinsics in each of the function.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerIntrin();

/*!
 * \brief Lower warp memory access to low-level device related function calls.
 * \return The pass.
 */
TVM_DLL Pass LowerWarpMemory();

/*!
 * \brief Lower attached storage access information on device.
 *
 * \note Run this pass after all storage access analysis finish.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerDeviceStorageAccessInfo();

/*!
 * \brief Combine context calls in the host function.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineContextCall();

/*!
 * \brief Narrow down PrimExpr datatype in stmt to target_bits.
 *
 * \param target_bits The target bits
 *
 * \note Run this pass after storage flatten.
 * \return The pass.
 */
TVM_DLL Pass NarrowDataType(int target_bits);

/*!
 * \brief Legalize bf16 typed Ops. Add a cast to fp32
 *   before Ops, then add a cast back to bf16.
 * \return The pass.
 */
TVM_DLL Pass BF16Legalize();

/*!
 * \brief Rewrite the pointer content type of arguments,
 *  as well as Alloc internal to the function to use
 *  the most frequently accessed type for load/store
 *  to avoid pointer casting in backend when possible.
 *
 * \return The pass.
 */
TVM_DLL Pass PointerValueTypeRewrite();

/*!
 * \brief Hoist loop-invariant IfThenElse nodes to
 * outside the elligible loops.
 *
 * \return The pass.
 */
TVM_DLL Pass HoistIfThenElse();

}  // namespace transform
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_H_
