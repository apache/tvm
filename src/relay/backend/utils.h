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
 * \file relay/backend/utils.h
 * \brief Utils function for backend
 */
#ifndef TVM_RELAY_BACKEND_UTILS_H_
#define TVM_RELAY_BACKEND_UTILS_H_

#include <dmlc/json.h>
#include <tvm/driver/driver_api.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/target/codegen.h>
#include <tvm/target/virtual_device.h>
#include <tvm/te/operation.h>
#include <tvm/tir/usmp/utils.h>

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../runtime/meta_data.h"
#include "../../target/metadata.h"
#include "tvm/runtime/ndarray.h"

namespace tvm {
namespace relay {

namespace tec {
class TECompiler;
}

namespace backend {
using Pass = tvm::transform::Pass;

/*! \brief Describes the type of kernel call emitted. */
enum CallType {
  /*!
   * \brief Emit PackedFunc calls bound just-in-time using TVMBackend* functions.
   *
   * When this type is selected, assumes all operators must be called via TVMFuncCall. Given the
   * implementation of TVMFuncCall in the C++ runtime, this in practice implies that those
   * functions are of type TVMBackendPackedCFunc.
   *
   * The following code is emitted at call sites to call a function named `func`:
   * void* func_ptr = TVMBackendGetFuncFromEnv("func");
   * TVMFuncCall(func_ptr, values, tcodes, num_args, ret_values, ret_tcodes)
   *
   * The arguments given to the tir::Call node are encoded into `values`, `tcodes`, and `num_args`
   * by LowerTVMBuiltin TIR transform.
   *
   * If `resource_handle` is passed to `func`, it is determined by TVMFuncCall (often,
   * `resource_handle` is registered with the C++ runtime to provide a `this` equivalent when
   * `func` is implemented in C).
   *
   * Compatible with both C++ and C runtimes, implemented with the C runtime only.
   */
  kPacked,  // Emit tir.call_packed and wrap all arguments in DLTensor.

  /*!
   * \brief Directly call a TVMBackendPackedCFunc named according to the tir::Call.
   *
   * When this type is selected, assumes all operators are implemented in functions of type
   * `TVMBackendPackedCFunc` and should be called directly. That is, presumes at the time of
   * downstream compilation that there is a symbol named after the 0th arg to tir::Call of
   * type `TVMBackendPackedCFunc`. This situation should occur when target_host == target.
   *
   * The following code is emitted at call sites to call a function named `func`:
   * func(values, tcodes, num_args, ret_values, ret_tcodes, resource_handle)
   *
   * The arguments given to the tir::Call node are encoded into `values`, `tcodes`, and `num_args`
   * by LowerTVMBuiltin TIR transform.
   *
   * `resource_handle` is encoded as the final argument to the tir::Call node. In practice, it is
   * always the device context parameter when not null. At present, the implementation does not
   * support forwarding device context parameters to CPacked.
   *
   * Compatible with the C runtime and C++ runtime (so long as target_host == target). Implemented
   * in the same scenarios.
   */
  kCPacked,  // Emit tir.call_cpacked and wrap all arguments in DLTensor.

  /*! \brief Directly call a function accepting the `data` arrays as args.
   *
   * When this type is selected, assumes all operaotrs are implemented in C functions whose
   * arguments are 1-to-1 with those in the tir::Call. DLTensor arguments are encoded as just the
   * `data` parameters (i.e. no DLTensor object is passed along).
   *
   * The following code is emitted at call sites to a function named `func`:
   * func(void* arg0, void* arg1, ..., void* argN) // no resource_handle
   * -or-
   * func(void* arg0, void* arg1, ..., void* argN, void* resource_handle) // with resource_handle
   *
   * `resource_handle` is encoded as the final argument to the tir::Call node. In practice, it is
   * always the device context parameter when not null.
   *
   * Compatible with the C runtime and C++ runtime (so long as target_host == target). Implemented
   * with the C runtime only.
   */
  kUnpacked,  // Emit tir.call_extern passing only the `data` part of DLTensors.
};

/*!
 * \brief Structure that can be optionally used by the executor codegen
 */
class ExecutorCodegenMetadataNode : public Object {
 public:
  /*! \brief input information for the main function */
  Array<tir::Var> inputs;
  /*! \brief input tensor type information */
  Array<TensorType> input_tensor_types;
  /*! \brief output information for the main function */
  Array<String> outputs;
  /*! \brief output tensor type information */
  Array<TensorType> output_tensor_types;
  /*! \brief pool information for the main function */
  Array<tir::Var> pools;
  /*! \brief device contexts information for the main function */
  Array<String> devices;
  /*! \brief the executor to be used to run the model */
  String executor = runtime::kTvmExecutorGraph;
  /*! \brief The external API (packed or c) in use */
  String interface_api;
  /*! \brief The internal API (packed or unpacked) in use */
  bool unpacked_api;
  /*! \brief Alginment of the workspace in bytes */
  Integer workspace_alignment;
  /*! \brief Alginment of the constants in bytes */
  Integer constant_alignment;
  /*! \brief the input var names that correspond to pool_inputs */
  Optional<Map<tir::Var, tir::usmp::AllocatedPoolInfo>> pool_inputs;
  /*! \brief the I/O tensor to PoolAllocations if any*/
  Map<String, tir::usmp::PoolAllocation> io_pool_allocations;

  String mod_name = "";

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("input_tensor_types", &input_tensor_types);
    v->Visit("outputs", &outputs);
    v->Visit("output_tensor_types", &output_tensor_types);
    v->Visit("pools", &pools);
    v->Visit("devices", &devices);
    v->Visit("executor", &executor);
    v->Visit("interface_api", &interface_api);
    v->Visit("unpacked_api", &unpacked_api);
    v->Visit("workspace_alignment", &workspace_alignment);
    v->Visit("constant_alignment", &constant_alignment);
    v->Visit("pool_inputs", &pool_inputs);
    v->Visit("io_pool_allocations", &io_pool_allocations);
    v->Visit("mod_name", &mod_name);
  }

  static constexpr const char* _type_key = "MetadataObj";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutorCodegenMetadataNode, Object);
};

/*!
 * \brief Managed reference to ExecutorCodegenMetadataNode.
 */
class ExecutorCodegenMetadata : public ObjectRef {
 public:
  TVM_DLL ExecutorCodegenMetadata(Array<tir::Var> inputs, Array<TensorType> input_tensor_types,
                                  Array<String> outputs, Array<TensorType> output_tensor_types,
                                  Array<tir::Var> pools, Array<String> devices, String executor,
                                  String mod_name, String interface_api = "packed",
                                  bool unpacked_api = false, Integer workspace_alignment = 16,
                                  Integer constant_alignment = 16,
                                  Map<tir::Var, tir::usmp::AllocatedPoolInfo> pool_inputs =
                                      Map<tir::Var, tir::usmp::AllocatedPoolInfo>(),
                                  Map<String, tir::usmp::PoolAllocation> io_pool_allocations = {});
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ExecutorCodegenMetadata, ObjectRef,
                                        ExecutorCodegenMetadataNode);
};

/*!
 * \brief The static storage information for each Tensor in the result of a Relay expression
 * (as per relay::FlattenTupleType).
 */
class StorageInfoNode : public Object {
 public:
  // TODO(mbs): Switch from struct-of-array to array-of-struct repr throughout.
  /*! \brief The set of storage ids where the expression is stored. */
  std::vector<int64_t> storage_ids;
  /* \brief The virtual devices these expressions are stored within. */
  std::vector<VirtualDevice> virtual_devices;
  /* \brief The sizes of each storage element, in bytes. */
  std::vector<int64_t> storage_sizes_in_bytes;

  // TODO(@jroesch): expose the fields
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.StorageInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(StorageInfoNode, Object);
};

/*! \brief The storage information for a single expression. */
class StorageInfo : public ObjectRef {
 public:
  StorageInfo(std::vector<int64_t> storage_ids, std::vector<VirtualDevice> virtual_devices,
              std::vector<int64_t> storage_sizes_in_bytes);
  TVM_DEFINE_OBJECT_REF_METHODS(StorageInfo, ObjectRef, StorageInfoNode);
};

/*!
 * \brief The result of static memory planning.
 */
class StaticMemoryPlanNode : public Object {
 public:
  Map<Expr, StorageInfo> expr_to_storage_info;

  void VisitAttrs(AttrVisitor* v) { v->Visit("expr_to_storage_info", &expr_to_storage_info); }

  static constexpr const char* _type_key = "relay.StaticMemoryPlan";
  TVM_DECLARE_FINAL_OBJECT_INFO(StaticMemoryPlanNode, Object);
};

/*! \brief The result of running static memory planning. */
class StaticMemoryPlan : public ObjectRef {
 public:
  explicit StaticMemoryPlan(Map<Expr, StorageInfo> expr_to_storage_info);
  TVM_DEFINE_OBJECT_REF_METHODS(StaticMemoryPlan, ObjectRef, StaticMemoryPlanNode);
};

struct FunctionInfoNode : public Object {
  Map<Target, Integer> workspace_sizes;
  Map<Target, Integer> io_sizes;
  Map<Target, Integer> constant_sizes;
  Map<Target, tir::PrimFunc> tir_primfuncs;
  Map<Target, Function> relay_primfuncs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("workspace_sizes", &workspace_sizes);
    v->Visit("io_sizes", &io_sizes);
    v->Visit("constant_sizes", &constant_sizes);
    v->Visit("tir_primfuncs", &tir_primfuncs);
    v->Visit("relay_primfuncs", &relay_primfuncs);
  }

  static constexpr const char* _type_key = "relay.backend.FunctionInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionInfoNode, Object);
};

class FunctionInfo : public ObjectRef {
 public:
  FunctionInfo(Map<Target, Integer> workspace_sizes, Map<Target, Integer> io_sizes,
               Map<Target, Integer> constant_sizes, Map<Target, tir::PrimFunc> tir_primfuncs,
               Map<Target, Function> relay_primfuncs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FunctionInfo, ObjectRef, FunctionInfoNode);
};

/*!
 * \brief Calculate the bytes of memory needed to hold a tensor of a given shape and data type.
 * \param shape The shape of the tensor
 * \param dtype The data type of the tensor
 */
size_t GetMemorySizeBytes(const Array<PrimExpr>& shape, const DataType& dtype);

/*!
 * \brief Calculate the storage required to store the type of relay.Expr
 *
 * \param func The relay expr for which the storage is calculated
 */
int64_t CalculateRelayExprSizeBytes(const Type& expr_type);

/*!
 *  \brief Executor generator artifacts. Those artifacts  are subsequently
 *  used by the relay build process.
 */
struct LoweredOutput {
  std::string graph_json;
  Map<Target, IRModule> lowered_funcs;
  Array<tvm::runtime::Module> external_mods;
  Map<String, FunctionInfo> function_metadata;
  /*!
   * \brief Map from constant names (allocated by the codegen as constants are encountered)
   * to the constant's value.
   */
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  ExecutorCodegenMetadata metadata;
};

/*!
 * \brief This class is needed to avoid a GCC 5 bug that prevents maps containing enums from being
 compiled. If i386 GCC version is increased, we can remove it.
 */
struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

/*!
 * \brief A helper to expand the params by adding the ones used in a given expression.
 */
struct ConstantUpdater : public ExprVisitor {
 public:
  ConstantUpdater(const std::string& symbol,
                  std::unordered_map<std::string, runtime::NDArray>* params)
      : symbol_(symbol), params_(params) {}

  void VisitExpr_(const ConstantNode* cn) final {
    std::string name = symbol_ + "_const_" + std::to_string(const_idx_++);
    VLOG(1) << "binding '" << name << "' to constant of type " << PrettyPrint(cn->checked_type());
    (*params_)[name] = cn->data;
  }

 private:
  int const_idx_{0};
  std::string symbol_;
  std::unordered_map<std::string, runtime::NDArray>* params_;
};

/*!
 * \brief A function to update the params with constants found in an external function.
 * \param func The function from which to get the constant params.
 * \param params The params to update with the constants.
 */
inline void UpdateConstants(BaseFunc func,
                            std::unordered_map<std::string, runtime::NDArray>* params) {
  VLOG_CONTEXT << "UpdateConstants";
  VLOG(1) << "updating constants for:" << std::endl << PrettyPrint(func);
  auto codegen = func->GetAttr<String>(attr::kCompiler);
  ICHECK(codegen.defined()) << "No external codegen is set";
  std::string codegen_name = codegen.value();
  const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  std::string symbol = std::string(name_node.value());
  std::string const_update_name = "relay.ext." + codegen_name + ".constant_updater";
  // Get the constant updater for the external codegen
  auto pf = tvm::runtime::Registry::Get(const_update_name);
  // If the backend hasn't registered a constant updater, use a default one
  if (pf == nullptr) {
    ConstantUpdater const_visit(symbol, params);
    const_visit(func);
  } else {
    Map<String, tvm::runtime::NDArray> constants = (*pf)(func, symbol);
    for (const auto& it : constants) {
      std::string const_name(it.first);
      // Constant names should begin this the compiler name (to avoid conflicts)
      ICHECK(const_name.find(codegen_name) == 0)
          << "External constant names must start with compiler name";
      (*params)[const_name] = it.second;
    }
  }
  for (const auto& pair : *params) {
    VLOG(1) << "Constants: " << pair.first << " = " << PrettyPrint(pair.second);
  }
}

/*!
 * \brief A simple wrapper around ExprFunctor for a single argument case.
 *  The result of visit is memoized.
 */
template <typename OutputType>
class MemoizedExprTranslator : public ::tvm::relay::ExprFunctor<OutputType(const Expr&)> {
  using BaseFunctor = ::tvm::relay::ExprFunctor<OutputType(const Expr&)>;

 public:
  /*! \brief virtual destructor */
  virtual ~MemoizedExprTranslator() {}

  /*!
   * \brief The memoized call.
   * \param n The expression node.
   * \return The result of the call
   */
  virtual OutputType VisitExpr(const Expr& n) {
    ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_[n] = res;
    return res;
  }

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, OutputType, ObjectPtrHash, ObjectPtrEqual> memo_;
};

/*!
 * \brief Get the Packed Func
 *
 * \param func_name
 * \return const PackedFunc*
 */
inline const PackedFunc* GetPackedFunc(const std::string& func_name) {
  return tvm::runtime::Registry::Get(func_name);
}

/*!
 * \brief Get a typed packed function.
 *
 * \param func_name
 * \return const PackedFunc*
 */
template <typename R, typename... Args>
inline const runtime::TypedPackedFunc<R(Args...)> GetTypedPackedFunc(const std::string& func_name) {
  auto* pf = GetPackedFunc(func_name);
  ICHECK(pf != nullptr) << "can not find packed function";
  return runtime::TypedPackedFunc<R(Args...)>(*pf);
}

/*!
 * \brief Extract shape from an IndexExpr array to std::vector<int64_t>
 *
 * \param shape The shape in Array
 * \return The converted shape in std::vector<int64_t>
 */
inline std::vector<int64_t> GetIntShape(const Array<IndexExpr>& shape) {
  std::vector<int64_t> ret;
  for (const auto& dim : shape) {
    const int64_t* pval = tir::as_const_int(dim);
    ret.push_back(pval ? *pval : -1);
  }
  return ret;
}

/*!
 * \brief Convert type to string
 *
 * \param typ
 * \return std::string string format of type
 */
inline std::string DType2String(const tvm::DataType dtype) {
  std::ostringstream os;
  if (dtype.is_float()) {
    os << "float";
  } else if (dtype.is_int()) {
    os << "int";
  } else if (dtype.is_uint()) {
    os << "uint";
  } else if (dtype.is_bfloat16()) {
    os << "bfloat";
  } else if ((*GetPackedFunc("runtime._datatype_get_type_registered"))(dtype.code())) {
    os << "custom["
       << (*GetPackedFunc("runtime._datatype_get_type_name"))(dtype.code()).operator std::string()
       << "]";
  } else {
    LOG(FATAL) << "Unknown type with code " << static_cast<unsigned>(dtype.code());
  }
  os << dtype.bits();
  return os.str();
}

/*!
 * \brief Bind params to function by using name
 * \param func Relay function
 * \param params params dict
 * \return relay::Function
 */
relay::Function BindParamsByName(relay::Function func,
                                 const std::unordered_map<std::string, runtime::NDArray>& params);

/*!
 * \brief Bind params to the main function in Relay module, using BindParamsByName
 * \param mod Relay module
 * \param params params dict
 */
void BindParamsInModule(IRModule mod,
                        const std::unordered_map<std::string, runtime::NDArray>& params);

void BindParamsInModule(IRModule mod, Map<String, runtime::NDArray> params);

/*!
 * \brief Extract the shape from a Relay tensor type.
 * \param type The provided type.
 * \return The extracted shape in a list.
 */
inline std::vector<int> GetShape(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  ICHECK(ttype) << "Expect TensorTypeNode";
  std::vector<int> shape;
  for (size_t i = 0; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImmNode>();
    ICHECK(val);
    shape.push_back(val->value);
  }
  return shape;
}

/*!
 * \brief Check if a call has the provided name.
 * \param call A Relay call node.
 * \param op_name The name of the expected call.
 * \return true if the call's name is equivalent to the given name. Otherwise,
 * false.
 */
inline bool IsOp(const CallNode* call, const std::string& op_name) {
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expects a single op.";
  Op op = GetRef<Op>(op_node);
  return op == Op::Get(op_name);
}

/*!
 * \brief Retrieve the "root" op nested inside a fused call, such as conv2d in relu(add(conv2d))
 * \param call A Relay call node. Typically nn.relu when called the first time.
 * \param depth The number of calls before the root op, counting from current_call.
 * \param expected_op_names The names of ops in this fused call. Example: {"nn.conv2d", "add",
 * "nn.relu"}
 * \return A CallNode corresponding to the root op, whose name is expected_op_names[0]
 */
inline const CallNode* GetRootCall(const CallNode* current_call, int depth,
                                   const std::vector<std::string>& expected_op_names) {
  ICHECK(current_call && depth >= 0 && static_cast<size_t>(depth) < expected_op_names.size() &&
         IsOp(current_call, expected_op_names[depth]));

  if (depth == 0) {
    return current_call;
  }

  ICHECK_GT(current_call->args.size(), 0);
  size_t valid_node_idx = 0;
  while (valid_node_idx < current_call->args.size() &&
         current_call->args[valid_node_idx].as<VarNode>()) {
    valid_node_idx++;
  }
  while (valid_node_idx < current_call->args.size() &&
         !(IsOp(current_call->args[valid_node_idx].as<CallNode>(), expected_op_names[depth - 1]))) {
    valid_node_idx++;
  }
  const auto* next_call = current_call->args[valid_node_idx].as<CallNode>();
  return GetRootCall(next_call, depth - 1, expected_op_names);
}

/*!
 * \brief Retrieve the "root" op nested inside a fused call, such as conv2d in relu(add(conv2d))
 * Unlike the previous definition, it does not verify operator names of intermediate nodes. Instead,
 * it recursively visit child nodes until it finds a call node with the given op_name.
 * \param call A Relay call node.
 * \param op_name The name of an op to look for, such as ""nn.conv2d".
 * \return A CallNode corresponding to the root op with the given op_name
 */
inline const CallNode* GetRootCall(const CallNode* current_call, const std::string& op_name) {
  if (current_call == nullptr) return nullptr;
  if (IsOp(current_call, op_name)) return current_call;

  ICHECK_GT(current_call->args.size(), 0);

  const auto* next_call = current_call->args[0].as<CallNode>();
  return GetRootCall(next_call, op_name);
}

/*!
 * \brief Retrieve the expected "root" op nested inside a fused call, such as conv2d in
 *        relu(add(conv2d))
 * \param call A Relay call node. Typically nn.relu when called the first time.
 * \param max_depth The maximum number of calls before the root op, counting from current_call.
 * \param op_name The name of expected "root" op in this fused call.
 * \return A CallNode corresponding to the root op
 */
inline const CallNode* GetRootCall(const CallNode* current_call, int max_depth,
                                   const std::string& op_name) {
  ICHECK(current_call && max_depth >= 0);

  if (max_depth == 0) {
    ICHECK(current_call && IsOp(current_call, op_name));
    return current_call;
  }
  if (IsOp(current_call, op_name)) {
    return current_call;
  }

  ICHECK_GT(current_call->args.size(), 0);

  size_t valid_node_idx = 0;
  while (valid_node_idx < current_call->args.size() &&
         current_call->args[valid_node_idx].as<VarNode>()) {
    valid_node_idx++;
  }

  const auto* next_call = current_call->args[valid_node_idx].as<CallNode>();
  return GetRootCall(next_call, max_depth - 1, op_name);
}

/*!
 * \brief Get the external symbol of the Relay function name.
 *
 * \param func The provided function.
 * \return An external symbol.
 */
inline std::string GetExtSymbol(const Function& func) {
  const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(name_node.defined()) << "Fail to retrieve external symbol.";
  return std::string(name_node.value());
}

/*!
 * \brief Return whether the auto scheduler is enabled in the pass context.
 */
inline bool IsAutoSchedulerEnabled() {
  return transform::PassContext::Current()
      ->GetConfig<Bool>("relay.backend.use_auto_scheduler", Bool(false))
      .value();
}

/*!
 * \brief Return whether the meta schedule is enabled in the pass context.
 */
inline bool IsMetaScheduleEnabled() {
  return transform::PassContext::Current()
      ->GetConfig<Bool>("relay.backend.use_meta_schedule", Bool(false))
      .value();
}

/*! \brief Consider MetaSchedule's dispatch option. */
inline int UseMetaScheduleDispatch() {
  return transform::PassContext::Current()
      ->GetConfig<Integer>("relay.backend.use_meta_schedule_dispatch", Integer(0))
      .value()
      ->value;
}
/*!
 * \brief Method in TECompiler to convert TE compute to scheduleable TIR
 * \param args The arguments of the TE compute
 * \param constants The constants used in AllocateConst
 * \return NullOpt if conversion fails; Otherwise the converted TIR
 * \note This method could be further used as a task filtering mechanism in task extraction
 */
using FTECompilerTIRConverter = runtime::TypedPackedFunc<  //
    Optional<tir::PrimFunc>(                               //
        const Array<te::Tensor>& args,                     //
        const Array<runtime::NDArray>& constants)>;

/*! \brief Return a task filter for AutoTIR according to `relay.backend.tir_converter` */
inline FTECompilerTIRConverter GetTIRConverter() {
  String name = transform::PassContext::Current()
                    ->GetConfig<String>("relay.backend.tir_converter", "default")
                    .value();
  const PackedFunc* f = runtime::Registry::Get("relay.backend.tir_converter." + name);
  ICHECK(f != nullptr) << "IndexError: Cannot find TIR converter: " << name;
  return FTECompilerTIRConverter(*f);
}

/*! \brief Converts a PrimFunc to IRModule. */
inline IRModule PrimFuncToIRModule(tir::PrimFunc f) {
  f = WithAttrs(f, Map<String, ObjectRef>{
                       {tvm::attr::kGlobalSymbol, String("main")},
                       {tvm::tir::attr::kNoAlias, Bool(1)},
                   });
  return IRModule({{GlobalVar("main"), f}});
}

/*!
 * \brief Get the sequence of Relay optimization passes based on backend type.
 * The prefix of the Relay passes almost overlaps between the vm and graph backend, with some slight
 * difference. This function unifies the shared optimization pass prefix between vm and graph
 * runtime, and returns the pass prefix given the backend type.
 *
 * \param is_homogeneous True if all primitives are to be executed on the same device and target.
 * \param is_vm True if passes are to be used for the vm executor.
 * \return An array of passes.
 */
Array<Pass> GetPassPrefix(bool is_homogeneous, bool is_vm);

/*! \brief Target hash function */
struct TargetStrHash {
  /*!
   * \brief Calculate the hash code of a Target based on the string value of the Target KIND.
   Note that this hash should NOT be used in new usecases, equality of targets based on their
   value is not well-defined.
   This will be removed when maps from Targets to IRModules are removed from the codebase.
   * \param target The Target to hash
   * \return String hash of the target
   */
  size_t operator()(const Target& target) const {
    std::string s(target->kind->name);
    return String::HashBytes(s.c_str(), s.size());
  }
};

/*! \brief Target equality function based on the string value of Target
Note that this equality function should NOT be used in new usecases, equality of targets based on
their value is not well-defined. This will be removed when maps from Targets to IRModules are
removed from the codebase.*/
struct TargetStrEqual {
  /*!
   * \brief Check if the two Targets are equal
   * \param target One Target
   * \param other_target The other Target
   * \return String equality of the targets
   */
  const bool operator()(const Target& target, const Target& other_target) const {
    TargetStrHash target_hash = TargetStrHash();
    return target_hash(target) == target_hash(other_target);
  }
};

/*!
 * \brief Convert a Map<Target, IRModule> to std::unordered_map<Target, IRmodule, TargetStrHash,
 * TargetStrEqual> Target equality is currently based on pointer equality, which is a problem since
 * we have a lot of Map<Target, IRModule> in the codebase. This function converts the map to a
 * version that is keyed based on string value of the Target instead. Note that once we remove
 * Map<Target, IRModule>, this function will be removed.
 * \param input_map The map to convert
 * \return The converted map
 */
std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual>
TargetModuleMapToTargetStrModuleMap(Map<Target, IRModule> input_map);

/*!
 * \brief Convert a std::unordered_map<Target, IRmodule, TargetStrHash, TargetStrEqual> to
 * Map<Target, IRModule> This function is a helper that undoes TargetModuleMapToTargetStr. Note that
 * once we remove Map<Target, IRModule>, this function will be removed.
 * \param input_map The map to convert
 * \return The converted map
 */
Map<Target, IRModule> TargetStrModuleMapToTargetModuleMap(
    std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual> input_map);

/*!
 * \brief Call "weight update callback" to communicate op weights seen during Relay module
 * lowering back to the auto scheduler.
 * Op weights refer to the number of times each distinct op/workload appears in a given module.
 * It is called "use_count" in TECompiler.
 * \param IRModule after lowering by LowerTEPass.
 */
void UpdateAutoSchedulerOpWeights(const IRModule& module);

/*!
 * \brief Extract shape from expr to vector<int64_t>
 *
 * \param shape
 * \return std::vector<int64_t>
 */
std::vector<int64_t> ShapeToJSON(tvm::Array<IndexExpr> shape);

}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_UTILS_H_
