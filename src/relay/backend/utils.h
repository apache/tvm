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
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/target/codegen.h>
#include <tvm/te/operation.h>

#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../runtime/meta_data.h"

namespace tvm {
namespace relay {
namespace backend {

/*!
 * \brief The static storage information produced by memory planning.
 */
class StorageInfoNode : public Object {
 public:
  /*! \brief The set of storage ids where the expression is stored. */
  std::vector<int64_t> storage_ids;
  /* \brief The type of "virtual devices" these expressions are stored on. */
  std::vector<DLDeviceType> device_types;
  /* \brief The sizes of each storage element. */
  std::vector<int64_t> storage_sizes_in_bytes;

  // TODO(@jroesch): expose the fields
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.StorageInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(StorageInfoNode, Object);
};

/*! \brief The storage information for a single expression. */
class StorageInfo : public ObjectRef {
 public:
  StorageInfo(std::vector<int64_t> storage_ids, std::vector<DLDeviceType> device_types,
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
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FunctionInfo, ObjectRef, FunctionInfoNode);
};

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
  Map<String, IRModule> lowered_funcs;
  Array<tvm::runtime::Module> external_mods;
  Map<String, FunctionInfo> function_metadata;
  std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>> params;
  runtime::Metadata metadata;
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
inline void UpdateConstants(Function func,
                            std::unordered_map<std::string, runtime::NDArray>* params) {
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
inline relay::Function BindParamsByName(
    relay::Function func, const std::unordered_map<std::string, runtime::NDArray>& params) {
  std::unordered_map<std::string, relay::Var> name_dict;
  std::unordered_set<relay::Var, ObjectPtrHash, ObjectPtrEqual> repeat_var;
  for (auto arg : func->params) {
    const auto& name = arg->name_hint();
    if (name_dict.count(name)) {
      repeat_var.insert(arg);
    } else {
      name_dict[name] = arg;
    }
  }

  std::unordered_map<relay::Var, Expr, ObjectPtrHash, ObjectPtrEqual> bind_dict;
  for (auto& kv : params) {
    if (name_dict.count(kv.first) == 0) {
      continue;
    }
    auto arg = name_dict.at(kv.first);
    if (repeat_var.count(arg)) {
      LOG(FATAL) << "Multiple args in the function have name " << kv.first;
    }
    bind_dict[arg] = Constant(kv.second);
  }
  Expr bound_expr = relay::Bind(func, bind_dict);
  Function ret = Downcast<Function>(bound_expr);
  ICHECK(ret.defined()) << "The returning type is expected to be a Relay Function."
                        << "\n";
  return ret;
}

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

  const auto* next_call = current_call->args[0].as<CallNode>();
  return GetRootCall(next_call, depth - 1, expected_op_names);
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
 * \brief Return whether the compile engine cache is disabled in the pass context.
 */
inline bool IsCompileEngineCacheDisabled() {
  return transform::PassContext::Current()
      ->GetConfig<Bool>("relay.backend.disable_compile_engine_cache", Bool(false))
      .value();
}

}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_UTILS_H_
