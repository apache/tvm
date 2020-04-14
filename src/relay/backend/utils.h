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
#include <tvm/tir/ir_pass.h>

#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relay {
namespace backend {

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
    CHECK(n.defined());
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
  std::unordered_map<Expr, OutputType, ObjectHash, ObjectEqual> memo_;
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
  CHECK(pf != nullptr) << "can not find packed function";
  return runtime::TypedPackedFunc<R(Args...)>(*pf);
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
  } else {
    LOG(FATAL) << "Unknown type";
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
  std::unordered_set<relay::Var, ObjectHash, ObjectEqual> repeat_var;
  for (auto arg : func->params) {
    const auto& name = arg->name_hint();
    if (name_dict.count(name)) {
      repeat_var.insert(arg);
    } else {
      name_dict[name] = arg;
    }
  }

  std::unordered_map<relay::Var, Expr, ObjectHash, ObjectEqual> bind_dict;
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
  CHECK(ret.defined()) << "The returning type is expected to be a Relay Function."
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
  CHECK(ttype) << "Expect TensorTypeNode";
  std::vector<int> shape;
  for (size_t i = 0; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImmNode>();
    CHECK(val);
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
  CHECK(op_node) << "Expects a single op.";
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
  CHECK(current_call && depth >= 0 && static_cast<size_t>(depth) < expected_op_names.size() &&
        IsOp(current_call, expected_op_names[depth]));

  if (depth == 0) {
    return current_call;
  }

  CHECK_GT(current_call->args.size(), 0);

  const auto* next_call = current_call->args[0].as<CallNode>();
  return GetRootCall(next_call, depth - 1, expected_op_names);
}

}  // namespace backend
}  // namespace relay
}  // namespace tvm


#endif  // TVM_RELAY_BACKEND_UTILS_H_
