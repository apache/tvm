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
 * \file tvm/tirx/function.h
 * \brief TIR Function.
 */
#ifndef TVM_TIRX_FUNCTION_H_
#define TVM_TIRX_FUNCTION_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/tensor.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <string>

namespace tvm {
namespace tirx {

/*!
 * \brief Primitive functions that contains TIR statements.
 *
 * The Function provides low-level code representation does not
 * automatically manage
 *
 * \sa Function
 */
class FunctionNode : public BaseFuncNode {
 public:
  /*! \brief Function parameters */
  ffi::Array<tvm::Var> params;
  /*! \brief The return type of the function. */
  Type ret_type = Type::Missing();
  /*! \brief The body of the function, absent for a declaration. */
  ffi::Optional<SeqStmt> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionNode>()
        .def_ro("params", &FunctionNode::params, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("ret_type", &FunctionNode::ret_type)
        .def_ro("body", &FunctionNode::body);
    refl::TypeAttrDef<FunctionNode>()
        .def(tvm::ffi::type_attr::kSEqual.name, &FunctionNode::SEqual)
        .def(tvm::ffi::type_attr::kSHash.name, &FunctionNode::SHash);
  }

  bool SEqual(const FunctionNode* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
    // `ty` is derived from the signature; compare its source fields directly.
    return equal(attrs, other->attrs, false, "attrs") &&
           equal(params, other->params, true, "params") &&
           equal(ret_type, other->ret_type, false, "ret_type") &&
           equal(body, other->body, false, "body");
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    int64_t hash_value = init_hash;
    hash_value = hash(attrs, hash_value, false);
    hash_value = hash(params, hash_value, true);
    hash_value = hash(ret_type, hash_value, false);
    hash_value = hash(body, hash_value, false);
    return hash_value;
  }

  /*!
   * \brief Return the derived function annotation of this function.
   *
   * \return The function type annotation.
   * \note The function type annotation of PrimExpr is
   *       directly derived from the Vars without the need of type inference.
   */
  TVM_TIRX_DLL FuncType func_type_annotation() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Function", FunctionNode, BaseFuncNode);
};

/*!
 * \brief Managed reference to FunctionNode.
 * \sa FunctionNode
 */
class Function : public BaseFunc {
 public:
  /*!
   * \brief Constructor
   *
   * \param params The parameters of the function.
   *
   * \param body The body of the function, or std::nullopt for a declaration.
   *
   * \param ret_type The return type of the function.
   *
   * \param attrs Additional function attributes.
   *
   * \param loc The location of this object in the source code.
   */
  TVM_TIRX_DLL Function(ffi::Array<tvm::Var> params, ffi::Optional<SeqStmt> body,
                        Type ret_type = VoidType(), DictAttrs attrs = DictAttrs(),
                        Location loc = UnknownLoc());

  explicit Function(ffi::ObjectPtr<FunctionNode> node) : BaseFunc(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Function, BaseFunc, FunctionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionNode);
};

/*!
 * \brief Renew variable definitions and remap their uses in a function.
 *
 * Definition regions follow the registered structural hooks. Free variables
 * and function attributes are preserved, and the function signature is rebuilt.
 *
 * \param func The function to renew.
 * \return The function with fresh variable definitions.
 */
TVM_TIRX_DLL Function RenewDef(Function func);

/*!
 * \brief Function specific attribute names.
 *
 * \sa tvm::attr
 */
namespace attr {

/*!
 * \brief Ordered launch-parameter tags for a device kernel.
 *
 * Type: ffi::Array<ffi::String>
 *
 * We call a device kernel launch function f using the following convention:
 *
 * Call(f,
 *      [arg1, arg2, ..., arg_n,
 *       work_size_1, work_size_2, ... work_size_m, dyn_shmem_size])
 *
 * Flag-only launch tags do not add packed operands.  The dynamic shared-memory
 * operand is present only when its value-bearing tag is listed.
 *
 * The list of kernel launch params indicates which additional
 * parameters will be provided to the ffi::Function by the calling
 * scope.
 *
 * - "threadIdx.x", "threadIdx.y", "threadIdx.z"
 *
 *   The extent of the thread count in x/y/z, to be used when
 *   launching the compute kernel on the device.  For example, the
 *   gridDimX/Y/Z parameters passed to cuLaunchKernel when launching a
 *   CUDA kernel, or the groupCountX/Y/Z parameters passed to
 *   vkCmdDispatch when dispatching a compute pipeline to Vulkan.
 *
 * - "blockIdx.x", "blockIdx.y", "blockIdx.z"
 *
 *   The extent of the block iterators, to be used when launching the
 *   compute kernel on the device.  For example, the blockDimX/Y/Z
 *   parameters passed to cuLaunchKernel when launching a CUDA kernel.
 *   For runtimes that do not require the block to be provided
 *   externally, this parameter is ignored.  For example, the
 *   spv::ExecutionModeLocalSize for SPIR-V shaders on Vulkan, where
 *   this parameter is defined in the shader.
 *
 * - tvm::runtime::launch_param::kUseDynamicSharedMemoryTag
 *
 *   The dynamic shared-memory byte count passed for this launch.
 *
 *   Defined as "tirx.use_dyn_shared_memory".
 *
 * - tvm::runtime::launch_param::kUseProgramaticDependentLaunch
 * - tvm::runtime::launch_param::kUseCooperativeLaunch
 * - tvm::runtime::launch_param::kUseRequiredBlockDimension
 *
 *   Flag-only launch attributes.  These tags add no packed operand.
 *
 * \sa tvm::CallingConv::kDeviceKernelLaunch
 */
inline constexpr ffi::reflection::AttrKey<ffi::Array<ffi::String>> kKernelLaunchParams{
    "tirx.kernel_launch_params"};

/*!
 * \brief Whether to set noalias rule on the function arguments.
 *
 * Type: bool
 */
inline constexpr ffi::reflection::AttrKey<bool> kNoAlias{"tirx.noalias"};

/*!
 * \brief Mark the function as the entry function of
 *        the final generated runtime module.
 *
 * Type: bool
 *
 * \note There can only be one entry function per module.
 */
inline constexpr ffi::reflection::AttrKey<bool> kIsEntryFunc{"tirx.is_entry_func"};

/*!
 * \brief Mark the function as the global function called from the host.
 *
 * Type: bool
 */
inline constexpr ffi::reflection::AttrKey<bool> kIsGlobalFunc{"tirx.is_global_func"};

/*!
 * \brief Mark the function as run on the host, mutually exclusive with kTarget.
 *
 * Type: bool
 */
inline constexpr ffi::reflection::AttrKey<bool> kIsHostFunc{"tirx.is_host_func"};

/*!
 * \brief Mark the kernel as persistent.
 */
inline constexpr ffi::reflection::AttrKey<bool> kPersistentKernel{"tirx.persistent_kernel"};

}  // namespace attr
}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_FUNCTION_H_
