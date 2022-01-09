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

#include <tvm/ir/function.h>
#include <tvm/runtime/ndarray.h>
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
  Array<tir::Var> params;
  /*! \brief The body of the function */
  tir::Stmt body;
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
   */
  Map<tir::Var, Buffer> buffer_map;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("body", &body);
    v->Visit("ret_type", &ret_type);
    v->Visit("buffer_map", &buffer_map);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimFuncNode* other, SEqualReducer equal) const {
    // visit params and buffer_map first as they contains defs.
    return equal.DefEqual(params, other->params) && equal(buffer_map, other->buffer_map) &&
           equal(ret_type, other->ret_type) && equal(body, other->body) &&
           equal(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(params);
    hash_reduce(buffer_map);
    hash_reduce(ret_type);
    hash_reduce(body);
    hash_reduce(attrs);
  }
  /*!
   * \brief Return the derived function annotation of this function.
   *
   * \return The function type annotation.
   * \note The function type annotation of PrimExpr is
   *       directly derived from the Vars without the need of type inference.
   */
  TVM_DLL FuncType func_type_annotation() const;

  static constexpr const char* _type_key = "tir.PrimFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncNode, BaseFuncNode);
};

/*!
 * \brief Managed reference to PrimFuncNode.
 * \sa PrimFuncNode
 */
class PrimFunc : public BaseFunc {
 public:
  /*!
   * \brief Constructor
   * \param params The parameters of the function.
   * \param body The body of the function.
   * \param ret_type The return type of the function.
   * \param buffer_map The buffer map for parameter buffer unpacking.
   * \param attrs Additional function attributes.
   * \param span The location of this object in the source code.
   */
  TVM_DLL PrimFunc(Array<tir::Var> params, Stmt body, Type ret_type = VoidType(),
                   Map<tir::Var, Buffer> buffer_map = Map<tir::Var, Buffer>(),
                   DictAttrs attrs = NullValue<DictAttrs>(), Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(PrimFunc, BaseFunc, PrimFuncNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PrimFuncNode);
};

/*!
 * \brief Describes one parameter that should be linked into the generated module.
 *
 * When parameters are to be linked in with generated code (i.e. on target_host-compatible
 * backends), Relay attaches instances of this object to a global TIR function. Code-generators
 * use the information contained in this node to include the parameter data in the generated
 * module.
 */
class LinkedParamNode : public Object {
 public:
  /*! \brief Unique numeric identifier used by runtimes to lookup this parameter. */
  int64_t id;

  /*! \brief Parameter data which should get linked into the final module. */
  ::tvm::runtime::NDArray param;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("id", &id);
    v->Visit("param", &param);
  }

  static constexpr const char* _type_key = "tir.LinkedParam";
  TVM_DECLARE_FINAL_OBJECT_INFO(LinkedParamNode, Object);
};

/*!
 * \brief Managed reference to LinkedParamNode.
 */
class LinkedParam : public ObjectRef {
 public:
  TVM_DLL LinkedParam(int64_t id, ::tvm::runtime::NDArray param);

  TVM_DEFINE_OBJECT_REF_METHODS(LinkedParam, ObjectRef, LinkedParamNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LinkedParamNode);
};

/*!
 * \brief Specialize parameters of PrimFunc.
 * \param func The PrimFunc to be specialized.
 * \param param_map The mapping from function params to the instance.
 * \return The new function with parameter specialized.
 * \note We can define a Meta TIR function with symbolic shape:
 *
 * \code
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
 * \code
 *  a, _, m, n = mem_copy.params
 *  func = mem_copy.specialize({a: tir.decl_buffer((16, 16))})
 *  # or
 *  func = mem_copy.specialize({n: 16, m: 16})
 * \endcode
 *
 * \code {.language-id}
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
PrimFunc Specialize(PrimFunc func, const Map<Var, ObjectRef>& param_map);

/*!
 * \brief PrimFunc specific attribute names.
 *
 * \sa tvm::attr
 */
namespace attr {
/*!
 * \brief List of thread IterVar that a DeviceLaunch function corresponds to.
 *
 * Type: Array<tir::IterVar>
 *
 * We call a device kernel launch function f using the following convention:
 *
 * Call(f,
 *      [arg1, arg2, ..., arg_n,
 *       work_size_1, work_size_2, ... work_size_m, dyn_shmem_size])
 *
 * Here n = len(arg), m = len(work_size) = len(device_thread_axis).
 *
 * When kDeviceUseDynSharedMemory is not set, dyn_shmem_size argument is omitted.
 *
 * The list of device_thread_axis indicates how can be bind the
 * work_size arguments to the corresponding threads.
 *
 * \sa tvm::CallingConv::kDeviceKernelLaunch
 */
constexpr const char* kDeviceThreadAxis = "tir.device_thread_axis";

/*!
 * \brief Whether or not use dynamic shared memory.
 *
 * Type: Integer
 */
constexpr const char* kDeviceUseDynSharedMemory = "tir.device_use_dyn_shared_memory";

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
 * \brief Parameters used in the module that should be linked by the codegen.
 *
 * Type: Map<String, LinkableParam>
 *
 * \note This should be present only on a function named
 *     tvm::target::packed_func::kLookupLinkedParam.
 */
constexpr const char* kLinkedParams = "tir.linked_params";

/*!
 * \brief Mark the function as the global function called from the host.
 *
 * Type: Integer
 */
constexpr const char* kIsGlobalFunc = "tir.is_global_func";

}  // namespace attr
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_FUNCTION_H_
