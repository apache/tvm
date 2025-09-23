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
 * \file tvm/tir/function.h
 * \brief TIR Function.
 */
#ifndef TVM_TIR_FUNCTION_H_
#define TVM_TIR_FUNCTION_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <string>

namespace tvm {
namespace tir {

/*!
 * \brief Primitive functions that contains TIR statements.
 *
 * The PrimFunc provides low-level code representation does not
 * automatically manage
 *
 * \sa PrimFunc
 */
class PrimFuncNode : public BaseFuncNode {
 public:
  /*! \brief Function parameters */
  ffi::Array<tir::Var> params;
  /*! \brief The return type of the function. */
  Type ret_type;
  /*!
   * \brief Maps some parameters to specific Buffer data structures.
   *
   *  buffer_map provides a way to express data structure's field and shape
   *  constraints. The provided information is used in the program analysis
   *  and the code generation.
   *
   *  - It defines the vars in the Buffer (m, n) in the cases below when
   *    they appears in the buffer_map for the first time.
   *  - When a var appears multiple times, they translate into runtime
   *    assertion to check the field constraint.
   *
   *  \code
   *
   *   # The corresponding fields of f are as follows
   *   #
   *   # - f.params = [a, b]
   *   # - f.buffer_map = {a: A, b: B}
   *   # - A = decl_buffer(shape=[m, n])
   *   # - B = decl_buffer(shape=[m, n])
   *
   *   def f(a, b):
   *       m, n = var(), var()
   *       A = bind_buffer(a, shape=[m, n])
   *       B = bind_buffer(b, shape=[m, n])
   *       # body
   *
   *  \endcode
   *
   *  buffer_map is a sugar to express:
   *  - Parameter unpacking: e.g. I can load a.shape[0] to get value of m
   *  - Constraint checking: a.shape[0] must equal b.shape[0] because they
   *    both corresponds to m.

   *  While we could have express parameter unpacking and constraint using
   *  normal statements, making buffer_map as first class citizen of PrimFunc
   *  will make program analysis much easier.
   *
   *  Prior to buffer flattening, which is performed FlattenBuffer for
   *  TIR-based schedules, these buffer objects are used directly in
   *  the body of the function.  After buffer flattening, these buffer
   *  objects remain unflattened for use in argument validation, but
   *  all usage in the body of the function is done through a
   *  flattened alias of the buffer.
   */
  ffi::Map<tir::Var, Buffer> buffer_map;
  /*! \brief The body of the function */
  tir::Stmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimFuncNode>()
        .def_ro("params", &PrimFuncNode::params, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("ret_type", &PrimFuncNode::ret_type)
        .def_ro("buffer_map", &PrimFuncNode::buffer_map)
        .def_ro("body", &PrimFuncNode::body);
  }

  /*!
   * \brief Return the derived function annotation of this function.
   *
   * \return The function type annotation.
   * \note The function type annotation of PrimExpr is
   *       directly derived from the Vars without the need of type inference.
   */
  TVM_DLL FuncType func_type_annotation() const;

  TVM_OBJECT_ENABLE_SCRIPT_PRINTER();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.PrimFunc", PrimFuncNode, BaseFuncNode);
};

/*!
 * \brief Managed reference to PrimFuncNode.
 * \sa PrimFuncNode
 */
class PrimFunc : public BaseFunc {
 public:
  /*!
   * \brief Constructor
   *
   * \param params The parameters of the function.
   *
   * \param body The body of the function.
   *
   * \param ret_type The return type of the function.
   *
   * \param buffer_map The buffer map for parameter buffer unpacking.
   * This contains buffer objects as they appear in the body of the
   * PrimFunc.  (e.g. a buffer of shape ``[1024]`` originally
   * generated as a tensor of shape ``[32, 32]``)
   *
   * \param attrs Additional function attributes.
   *
   * \param span The location of this object in the source code.
   */
  TVM_DLL PrimFunc(ffi::Array<tir::Var> params, Stmt body, Type ret_type = VoidType(),
                   ffi::Map<tir::Var, Buffer> buffer_map = ffi::Map<tir::Var, Buffer>(),
                   DictAttrs attrs = DictAttrs(), Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimFunc, BaseFunc, PrimFuncNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PrimFuncNode);
};

/*!
 * \brief Tensor intrinsics for tensorization
 */
class TensorIntrinNode : public Object {
 public:
  /*! \brief The function to describe the computation. */
  PrimFunc desc;
  /*! \brief The function of the implementation for the execution. */
  PrimFunc impl;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorIntrinNode>()
        .def_ro("desc", &TensorIntrinNode::desc)
        .def_ro("impl", &TensorIntrinNode::impl);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.TensorIntrin", TensorIntrinNode, Object);
};

/*!
 * \brief Managed reference to TensorIntrinNode.
 */
class TensorIntrin : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param desc The function to describe the computation.
   * \param impl The function of the implementation for the execution.
   */
  TVM_DLL explicit TensorIntrin(PrimFunc desc, PrimFunc impl);

  /*!
   * \brief Create and register a TensorIntrin. After registration, the TensorIntrin can be looked
   * up with its name.
   * \param name The name of the TensorIntrin to register
   * \param intrin The TensorIntrin to register.
   * \param override Whether override existing intrinsic.
   * \throws This method throws an exception if the TensorIntrin with the specified name already
   *         exists.
   */
  TVM_DLL static void Register(ffi::String name, TensorIntrin intrin, bool override = false);

  /*!
   * \brief Look up TensorIntrin by name. Raises an exception if not found.
   * \param name The name of the TensorIntrin.
   * \param allow_missing Whether to allow missing tensor intrin. If false, an exception is raised
   *    if the tensor intrin is not found.
   * \return The TensorIntrin with the specified name.
   * \throws This method throws an exception if the TensorIntrin does not exist and allow_missing is
   * false.
   */
  TVM_DLL static ffi::Optional<TensorIntrin> Get(ffi::String name, bool allow_missing = false);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorIntrin, ObjectRef, TensorIntrinNode);
};

/*!
 * \brief Specialize parameters of PrimFunc.
 * \param func The PrimFunc to be specialized.
 * \param param_map The mapping from function params to the instance.
 * \return The new function with parameter specialized.
 * \note We can define a Meta TIR function with symbolic shape:
 *
 * \code{.py}
 *  @T.prim_func
 *  def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32) -> None:
 *      A = T.match_buffer(a, (m, n), "float32")
 *      B = T.match_buffer(b, (m, n), "float32")
 *      for i, j in T.grid(m, n):
 *          with T.block():
 *              vi, vj = T.axis.remap("SS", [i, j])
 *              B[vi, vj] = A[vi, vj]
 * \endcode
 *
 * Then we can make it specialized with given shapes or buffers.
 *
 * \code{.py}
 *  a, _, m, n = mem_copy.params
 *  func = mem_copy.specialize({a: tir.decl_buffer((16, 16))})
 *  # or
 *  func = mem_copy.specialize({n: 16, m: 16})
 * \endcode
 *
 * \code{.py}
 *  @T.prim_func
 *  def mem_copy_16_16(a: T.handle, b: T.handle) -> None:
 *      A = T.match_buffer(a, (16, 16), "float32")
 *      B = T.match_buffer(b, (16, 16), "float32")
 *      for i, j in T.grid(16, 16):
 *          with T.block():
 *              vi, vj = T.axis.remap("SS", [i, j])
 *              B[vi, vj] = A[vi, vj]
 * \endcode
 */
PrimFunc Specialize(PrimFunc func, const ffi::Map<Var, ffi::Variant<Buffer, PrimExpr>>& param_map);

/*!
 * \brief PrimFunc specific attribute names.
 *
 * \sa tvm::attr
 */
namespace attr {

/*!
 * \brief List of thread IterVar that a DeviceLaunch function corresponds to.
 *
 * Type: ffi::Array<ffi::String>
 *
 * We call a device kernel launch function f using the following convention:
 *
 * Call(f,
 *      [arg1, arg2, ..., arg_n,
 *       work_size_1, work_size_2, ... work_size_m, dyn_shmem_size])
 *
 * Here n = len(arg), m = len(work_size) = len(launch_params)-1.
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
 *   The size of the shared memory that may be allocated internally by
 *   the kernel.  For example, exposed as the
 *   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES attribute in
 *   cuda.
 *
 *   Defined as "tir.use_dyn_shared_memory".
 *
 * \sa tvm::CallingConv::kDeviceKernelLaunch
 */
constexpr const char* kKernelLaunchParams = "tir.kernel_launch_params";

/*!
 * \brief Whether to set noalias rule on the function arguments.
 *
 * Type: Integer
 */
constexpr const char* kNoAlias = "tir.noalias";

/*!
 * \brief Mark the function as the entry function of
 *        the final generated runtime module.
 *
 * Type: Integer
 *
 * \note There can only be one entry function per module.
 */
constexpr const char* kIsEntryFunc = "tir.is_entry_func";

/*!
 * \brief Mark the function as the global function called from the host.
 *
 * Type: Integer
 */
constexpr const char* kIsGlobalFunc = "tir.is_global_func";

/*!
 * \brief Mark the function as run on the host, mutually exclusive with kTarget.
 *
 * Type: Integer
 */
constexpr const char* kIsHostFunc = "tir.is_host_func";

/*!
 * \brief Mark the function as scheduled, so the default schedule will pass will skip it.
 *
 * Type: Integer
 */
constexpr const char* kIsScheduled = "tir.is_scheduled";

}  // namespace attr
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_FUNCTION_H_
