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

#ifndef TVM_RELAY_BACKEND_CONTRIB_DNNL_CODEGEN_TOOLS_H_
#define TVM_RELAY_BACKEND_CONTRIB_DNNL_CODEGEN_TOOLS_H_

#include <tvm/ir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/interpreter.h>

#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../op/make_op.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

namespace details {

template <typename T>
DataType make_dtype();

template <>
DataType make_dtype<int32_t>() {
  return DataType::Int(32);
}
template <>
DataType make_dtype<uint32_t>() {
  return DataType::UInt(32);
}
template <>
DataType make_dtype<int8_t>() {
  return DataType::Int(8);
}
template <>
DataType make_dtype<uint8_t>() {
  return DataType::UInt(8);
}
template <>
DataType make_dtype<float>() {
  return DataType::Float(32);
}

}  // namespace details

/*!
 * @brief Infer type for provided expression
 */
Expr InferType(const Expr& expr) {
  auto mod_and_global = IRModule::FromExprInContext(expr, {}, {}, {});
  auto mod = transform::InferType()(mod_and_global.first);
  auto inferred = Downcast<Function>(mod->Lookup(mod_and_global.second->name_hint));
  return inferred->body;
}

/*!
 * @brief Evaluate expression if possible
 *
 * Transformation rules:
 *   Empty expr -> Empty expr
 *   Constant expr -> ConstantNode with corresponding values
 *   All other expr -> original expr
 *
 * @param expr original expression
 * @return resulting expr. Corresponding ConstantNode or original expr
 */
Expr EvalExpr(Expr expr) {
  if (!expr.defined()) return expr;

  Device dev{kDLCPU, 0};
  Target target = Target("llvm");

  With<transform::PassContext> fresh_build_ctx(transform::PassContext::Create());
  auto res = Eval(expr, {}, {}, dev, target);

  if (res->IsInstance<runtime::NDArray::ContainerType>()) {
    auto nd_array = Downcast<runtime::NDArray>(res);
    return InferType(Constant(nd_array));
  } else {
    LOG(ERROR) << "Unexpected object type";
  }
  return {};
}

/*!
 * @brief Evaluate shape of resulting tensor for provided expression
 * @param exp expression to evaluate result shape
 * @return shape of tensor
 */
static std::vector<uint32_t> shape_of(const Expr& exp) {
  auto typed_exp = InferType(exp);
  auto tt = typed_exp->checked_type().as<TensorTypeNode>();

  ICHECK(tt) << "Expr has none tensor type";

  std::vector<uint32_t> res;
  for (const auto d : tt->shape) {
    auto i_d = d.as<IntImmNode>();
    ICHECK(i_d);
    res.push_back(i_d->value);
  }
  return res;
}

/*!
 * @brief Evaluate shape of resulting tensor
 * @param exp expression to evaluate
 * @return resulting data type
 */
static DataType dtype_of(const Expr& exp) {
  auto typed_exp = InferType(exp);
  auto tt = typed_exp->checked_type().as<TensorTypeNode>();

  ICHECK(tt) << "Expr is not tensor type";
  return tt->dtype;
}

static bool is_scalar(const Expr& exp) {
  const Expr typed_exp = exp.defined() ? exp : InferType(exp);
  const auto* tt = typed_exp->type_as<TensorTypeNode>();
  ICHECK(tt) << "Expression is not Tensor producer";
  return tt->shape.size() == 0;
}

static bool is_const(const Expr& exp) { return exp->IsInstance<ConstantNode>(); }

template <typename T>
static bool is_const_scalar_eq(const Expr& exp, T val) {
  if (details::make_dtype<T>() != dtype_of(exp)) return false;
  if (const auto* constant = exp.as<ConstantNode>()) {
    if (constant->data->ndim == 0) {
      return *static_cast<T*>(constant->data->data) == val;
    }
  }
  return false;
}

Constant constant(int val) {
  auto value = runtime::NDArray::Empty({}, DataType::Int(32), {kDLCPU, 0});
  value.CopyFromBytes(&val, sizeof(val));
  return Constant(value);
}

Constant constant(float val) {
  auto value = runtime::NDArray::Empty({}, DataType::Float(32), {kDLCPU, 0});
  value.CopyFromBytes(&val, sizeof(val));
  return Constant(value);
}

/*!
 * @brief Check if expr produce a tensor with broadcast value.
 * If yes return corresponding scalar value otherwise return the same expr.
 */
Expr collapse_to_scalar(const Expr& exp) {
  const Expr const_exp = is_const(exp) ? exp : EvalExpr(exp);
  if (is_scalar(const_exp)) return const_exp;

  if (const auto* const_node = const_exp.as<ConstantNode>()) {
    auto ptr = static_cast<int*>(const_node->data->data);
    auto size = const_node->data->shape[0];
    bool is_same = true;
    for (int i = 0; i < size; i++) {
      is_same &= ptr[i] == ptr[0];
    }
    if (is_same) {
      return EvalExpr(constant(ptr[0]));
    }
  }
  return exp;
}

template <typename T>
static Expr cast(const Expr& that) {
  return MakeCast(that, details::make_dtype<T>());
}

static Expr squeeze(const Expr& exp) {
  const Expr typed_exp = exp.defined() ? exp : InferType(exp);
  const auto* tt = typed_exp->type_as<TensorTypeNode>();
  ICHECK(tt) << "Expression is not Tensor producer";
  // Empty list doesn't work. Have to specify it manually
  Array<Integer> axis_to_squeeze;
  for (size_t i = 0; i < tt->shape.size(); i++)
    if (tt->shape[i].as<IntImmNode>()->value == 1) axis_to_squeeze.push_back(i);

  return MakeSqueeze(exp, axis_to_squeeze);
}

static Expr permute(const Expr& exp, const Array<Integer>& perm) {
  return MakeTranspose(exp, perm);
}

static Expr broadcast(const Expr& exp, const Array<Integer>& shape) {
  return MakeBroadCastTo(exp, shape);
}

static Array<Integer> permutation(const std::string& from, const std::string& to) {
  Array<Integer> perm;
  for (const auto& c : to) {
    auto found = from.find(c);
    ICHECK_NE(found, std::string::npos);
    perm.push_back(found);
  }
  return perm;
}

/*!
 * @brief Helper namespace. Introduce elementwise arithmetic operations for expressions
 *
 * Broadcast semantic is included(forward and backward). If result tensor is a broadcast value it
 * may be collapsed into scalar.
 */
namespace tensor_arithmetic {

Expr operator+(const Expr& lhs, const Expr& rhs) {
  if (is_const_scalar_eq(lhs, 0) || is_const_scalar_eq(lhs, 0.0f)) return rhs;
  if (is_const_scalar_eq(rhs, 0) || is_const_scalar_eq(rhs, 0.0f)) return lhs;

  static const Op& op = Op::Get("add");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

Expr operator-(const Expr& that) {
  static const Op& op = Op::Get("negative");
  return Call(op, {that}, Attrs(), {});
}

Expr operator-(const Expr& lhs, const Expr& rhs) {
  if (is_const_scalar_eq(lhs, 0) || is_const_scalar_eq(lhs, 0.0f)) return -rhs;
  if (is_const_scalar_eq(rhs, 0) || is_const_scalar_eq(rhs, 0.0f)) return lhs;

  static const Op& op = Op::Get("subtract");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

Expr operator*(const Expr& lhs, const Expr& rhs) {
  if (is_const_scalar_eq(lhs, 1) || is_const_scalar_eq(lhs, 1.0f)) return rhs;
  if (is_const_scalar_eq(rhs, 1) || is_const_scalar_eq(rhs, 1.0f)) return lhs;
  if (is_const_scalar_eq(lhs, 0) || is_const_scalar_eq(lhs, 0.0f)) return constant(0.0f);
  if (is_const_scalar_eq(rhs, 0) || is_const_scalar_eq(rhs, 0.0f)) return constant(0.0f);

  static const Op& op = Op::Get("multiply");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

Expr operator/(const Expr& lhs, const Expr& rhs) {
  if (is_const_scalar_eq(rhs, 1) || is_const_scalar_eq(rhs, 1.0f)) return lhs;

  static const Op& op = Op::Get("divide");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

}  // namespace tensor_arithmetic

/*!
 * @brief Graph linearizator. Construct sequence of CallNode objects in post dfs order.
 * Helpful to check existence of some op in Function expr. And search it by name.
 */
class OpSeq : public ExprVisitor {
 public:
  struct Layer {
    const CallNode* call_node_ = nullptr;
    std::vector<relay::Expr> extern_args_ = {};

    operator bool() const { return call_node_ != nullptr; }
  };

  /** return op descriptor for provided name, or empty layer if not exists */
  const Layer& getOpLayer(const std::string& name) const {
    static Layer empty;

    auto found = std::find_if(layers_.begin(), layers_.end(), [&name](auto& l) {
      return l.call_node_->op.template as<OpNode>()->name == name;
    });

    const auto& res = (found == layers_.end()) ? empty : *found;
    return res;
  }

  /** return list of call node names if post dfs order */
  std::vector<std::string> getOpNames() const {
    std::vector<std::string> res;
    for (auto& l : layers_) res.push_back(l.call_node_->op.as<OpNode>()->name);
    return res;
  }

 protected:
  void VisitExpr_(const CallNode* cn) final {
    ExprVisitor::VisitExpr_(cn);

    Layer res{cn};
    for (const auto& arg : cn->args) {
      if (arg->IsInstance<relay::VarNode>() || arg->IsInstance<relay::ConstantNode>())
        res.extern_args_.push_back(arg);
    }
    layers_.push_back(res);
  }
  std::vector<Layer> layers_;
};

class OpAttrMapExtractor : public AttrVisitor {
 public:
  OpAttrMapExtractor() {}

  const std::unordered_map<std::string, dmlc::any>& get() { return attr_map; }

  template <typename T = double, typename = std::enable_if_t<std::is_floating_point<T>::value>>
  std::string Fp2String(const T value) {
    std::ostringstream out;
    out.precision(std::numeric_limits<T>::max_digits10);
    out << value;
    return out.str();
  }

  void SetNodeAttr(const char* key, const std::vector<std::string>& value) {
    std::vector<dmlc::any> attr;
    attr.emplace_back(value);
    attr_map[key] = dmlc::any{attr};
  }

  void Visit(const char* key, double* value) final { SetNodeAttr(key, {Fp2String(*value)}); }

  void Visit(const char* key, int64_t* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, uint64_t* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, int* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, bool* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, std::string* value) final { SetNodeAttr(key, {*value}); }

  void Visit(const char* key, DataType* value) final {
    if (!value->is_void()) {
      SetNodeAttr(key, {runtime::DLDataType2String(*value)});
    } else {
      SetNodeAttr(key, {""});
    }
  }

  void Visit(const char* key, runtime::ObjectRef* value) final {
    if (const auto* an = (*value).as<ArrayNode>()) {
      std::vector<std::string> attr;
      for (size_t i = 0; i < an->size(); ++i) {
        if (const auto* im = (*an)[i].as<IntImmNode>()) {
          attr.push_back(std::to_string(im->value));
        } else if (const auto* fm = (*an)[i].as<FloatImmNode>()) {
          attr.push_back(Fp2String(fm->value));
        } else if (const auto* str = (*an)[i].as<StringObj>()) {
          String s = GetRef<String>(str);
          attr.push_back(s);
        } else {
          LOG(FATAL) << "Not supported type: " << (*an)[i]->GetTypeKey();
        }
      }
      SetNodeAttr(key, attr);
    } else if (!(*value).defined()) {  // Skip NullValue
      SetNodeAttr(key, std::vector<std::string>{""});
    } else if (const auto* im = (*value).as<IntImmNode>()) {
      SetNodeAttr(key, std::vector<std::string>{std::to_string(im->value)});
    } else if (const auto* fm = (*value).as<FloatImmNode>()) {
      SetNodeAttr(key, std::vector<std::string>{Fp2String(fm->value)});
    } else if (const auto* str = (*value).as<StringObj>()) {
      String s = GetRef<String>(str);
      SetNodeAttr(key, std::vector<std::string>{s});
    } else {
      LOG(FATAL) << "Not yet supported type: " << (*value)->GetTypeKey() << ": " << *value;
    }
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "NDArray is not allowed in op attribute";
  }

  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "void pointer is not allowed in op attribute";
  }

  void Extract(Object* node) {
    if (node) {
      reflection_->VisitAttrs(node, this);
    }
  }

 private:
  std::unordered_map<std::string, dmlc::any> attr_map;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();
};

/*!
 * @brief Helper function to extract attributes as collection of dmlc objects
 *
 * @param node node to extract attrs
 * @return resulting collection of attributes
 */
std::unordered_map<std::string, dmlc::any> extractAttrs(const CallNode* node) {
  OpAttrMapExtractor extractor;
  const Object* call_attr = node->attrs.get();
  extractor.Extract(const_cast<Object*>(call_attr));
  return extractor.get();
}

/*!
 * @brief Converter attribute to dmlc acceptable format
 *
 * @tparam T type of value (auto deduction)
 * @param val value to convert
 * @return resulting dmlc object
 */
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
dmlc::any dmlc_attr(const T& val) {
  std::vector<dmlc::any> attr;
  attr.emplace_back(std::vector<std::string>{std::to_string(val)});
  return dmlc::any{attr};
}

template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, bool> = true>
dmlc::any dmlc_attr(const T& val) {
  std::vector<dmlc::any> attr;
  attr.emplace_back(std::vector<std::string>{val});
  return dmlc::any{attr};
}

template <typename T,
          std::enable_if_t<std::is_same<T, std::vector<std::string>>::value, bool> = true>
dmlc::any dmlc_attr(const T& val) {
  std::vector<dmlc::any> attr;
  attr.emplace_back(val);
  return dmlc::any{attr};
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_DNNL_CODEGEN_TOOLS_H_
