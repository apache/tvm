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
 * \file tvm/relay/op.h
 * \brief Primitive operator definition.
 */
#ifndef TVM_RELAY_OP_H_
#define TVM_RELAY_OP_H_

#include <functional>
#include <limits>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "base.h"
#include "expr.h"
#include "type.h"

namespace tvm {
namespace relay {

// forward declare name.
template <typename ValueType>
class OpMap;
class GenericOpMap;
class OpRegistry;

/*!
 * \brief Node container of operator structure.
 */
class OpNode : public relay::ExprNode {
 public:
  /*! \brief name of the operator */
  std::string name;
  /*! \brief the type of the operator */
  mutable FuncType op_type;
  /*!
   * \brief detailed description of the operator
   *  This can be used to generate docstring automatically for the operator.
   */
  std::string description;
  /* \brief Information of input arguments to the operator */
  Array<AttrFieldInfo> arguments;
  /*!
   * \brief The type key of the attribute field
   *  This can be empty, in which case it defaults to anything.
   */
  std::string attrs_type_key;
  /*!
   * \brief attribute type index,
   * this field varies in each run and is not exposed to frontend.
   */
  uint32_t attrs_type_index{0};
  /*!
   * \brief number of input arguments to the operator,
   * -1 means it is variable length
   */
  int32_t num_inputs = -1;
  /*!
   * \brief support level of the operator,
   *  The lower the more priority it contains.
   *  This is in analogies to BLAS levels.
   */
  int32_t support_level = 10;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("op_type", &op_type);
    v->Visit("description", &description);
    v->Visit("arguments", &arguments);
    v->Visit("attrs_type_key", &attrs_type_key);
    v->Visit("num_inputs", &num_inputs);
    v->Visit("support_level", &support_level);
  }

  /*!
   * \brief Check that if current op is a "primtive operator".
   * That is the arguments are all type variables, and there is a single
   * type relation applied to the input and output types.
   */
  bool IsPrimitiveOp() const {
    if (is_primitive_ != -1) return is_primitive_ != 0;
    is_primitive_ = this->IsPrimitiveOp_() ? 1 : 0;
    return is_primitive_ != 0;
  }

  static constexpr const char* _type_key = "relay.Op";
  TVM_DECLARE_NODE_TYPE_INFO(OpNode, ExprNode);

 private:
  // friend class
  friend class GenericOpMap;
  friend class OpRegistry;
  friend bool IsPrimitiveOp(const Expr&);
  // Program internal unique index of operator.
  // Used to help index the program.
  uint32_t index_{0};
  // whether this is a primitive op. -1 means unknown.
  mutable int is_primitive_{-1};
  // Internal function to compute if it is primitive op
  bool IsPrimitiveOp_() const {
    const auto& fn_ty = this->op_type;
    if (fn_ty->type_constraints.size() != 1) return false;
    const TypeRelationNode* rel = fn_ty->type_constraints[0].as<TypeRelationNode>();
    if (rel == nullptr) return false;
    // validate if the type parameter matches up
    for (size_t i = 0; i < fn_ty->type_params.size(); ++i) {
      if (!fn_ty->type_params[i].same_as(rel->args[i])) return false;
    }
    return true;
  }
};

/*!
 * \brief Operator reference class.
 */
class Op : public relay::Expr {
 public:
  /*! \brief default constructor  */
  Op() {}
  /*! \brief constructor from node pointer */
  explicit Op(NodePtr<Node> n) : Expr(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const OpNode* operator->() const;
  /*!
   * \brief Get additional registered attribute about operators.
   *  If nothing has been registered, an empty OpMap will be returned.
   * \param attr_name The name of the attribute.
   * \return An OpMap of specified attr_name.
   * \tparam ValueType The type of the attribute.
   */
  template <typename ValueType>
  inline static OpMap<ValueType> GetAttr(const std::string& attr_name);
  /*!
   * \brief Get an Op for a given operator name.
   *  Will raise an error if the op has not been registered.
   * \param op_name Name of the operator.
   * \return Pointer to a Op, valid throughout program lifetime.
   */
  TVM_DLL static const Op& Get(const std::string& op_name);

  /*! \brief specify container node */
  using ContainerType = OpNode;

 private:
  /*!
   * \brief Get generic attrmap given attr name
   * \param key The attribute key
   * \return reference to GenericOpMap
   */
  TVM_DLL static const GenericOpMap& GetGenericAttr(const std::string& key);
};

/*! \brief Helper structure to register operators */
class OpRegistry {
 public:
  /*! \return the operator */
  const Op& op() const { return op_; }
  /*!
   * \brief setter function during registration
   *  Set the description of operator
   * \param descr the description string.
   * \return reference to self.
   */
  inline OpRegistry& describe(const std::string& descr);  // NOLINT(*)
  /*!
   * \brief Add argument information to the function.
   * \param name Name of the argument.
   * \param type Type of the argument.
   * \param description Description of the argument.
   * \return reference to self.
   */
  inline OpRegistry& add_argument(const std::string& name,
                                  const std::string& type,
                                  const std::string& description);
  /*!
   * \brief Attach the type function corresponding to the return type.
   * \param rel_name The type relation name to register.
   * \param type_rel_func The backing relation function which can solve an arbitrary
   * relation on variables.
   * \return reference to self.
   */
  inline OpRegistry& add_type_rel(
      const std::string& rel_name,
      runtime::TypedPackedFunc<bool(const Array<Type>&,
                                    int,
                                    const Attrs&,
                                    const TypeReporter&)> type_rel_func);
  /*!
   * \brief Set the type key of attributes.
   * \param type_key The type of of the attrs field.
   * \return reference to self.
   */
  inline OpRegistry& set_attrs_type_key(const std::string& type_key);
  /*!
   * \brief Set the num_inputs
   * \param n The number of inputs to be set.
   * \return reference to self.
   */
  inline OpRegistry& set_num_inputs(int32_t n);  // NOLINT(*)
  /*!
   * \brief Set the support level of op.
   * \param level The support level.
   * \return reference to self.
   */
  inline OpRegistry& set_support_level(int32_t level);  // NOLINT(*)
  /*!
   * \brief Register additional attributes to operator.
   * \param attr_name The name of the attribute.
   * \param value The value to be set.
   * \param plevel The priority level of this set,
   *  an higher priority level attribute
   *  will replace lower priority level attribute.
   *  Must be bigger than 0.
   *
   *  Cannot set with same plevel twice in the code.
   *
   * \tparam ValueType The type of the value to be set.
   */
  template <typename ValueType>
  inline OpRegistry& set_attr(const std::string& attr_name,  // NOLINT(*)
                              const ValueType& value, int plevel = 10);

  // set the name of the op to be the same as registry
  inline OpRegistry& set_name() {  // NOLINT(*)
    if (get()->name.length() == 0) {
      get()->name = name;
    }
    return *this;
  }
  /*! \return The global single registry */
  TVM_DLL static ::dmlc::Registry<OpRegistry>* Registry();

 private:
  friend class ::dmlc::Registry<OpRegistry>;
  // the name
  std::string name;
  /*! \brief The operator */
  Op op_;
  // private constructor
  TVM_DLL OpRegistry();
  // return internal pointer to op.
  inline OpNode* get();
  // update the attribute OpMap
  TVM_DLL void UpdateAttr(const std::string& key, TVMRetValue value,
                          int plevel);
};

/*!
 * \brief Generic map to store additional information of Op.
 */
class GenericOpMap {
 public:
  /*!
   * \brief Check if the map has op as key.
   * \param op The key to the map
   * \return 1 if op is contained in map, 0 otherwise.
   */
  inline int count(const Op& op) const;
  /*!
   * \brief get the corresponding value element at op
   * \param op The key to the map
   * \return the const reference to the content value.
   */
  inline const TVMRetValue& operator[](const Op& op) const;
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param op The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   * \tparam ValueType The content value type.
   */
  template <typename ValueType>
  inline ValueType get(const Op& op, ValueType def_value) const;
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param expr The key to the map
   * \param def_value The default value when the key does not exist
   *         or if expr is not an Op.
   * \return the const reference to the content value.
   * \tparam ValueType The content value type.
   */
  template <typename ValueType>
  inline ValueType get(const Expr& expr, ValueType def_value) const;

 private:
  friend class OpRegistry;
  // the attribute field.
  std::string attr_name_;
  // internal data
  std::vector<std::pair<TVMRetValue, int> > data_;
  // The value
  GenericOpMap() = default;
};

/*!
 * \brief Map<Op,ValueType> used to store meta-information about Op.
 * \tparam ValueType The type of the value stored in map.
 */
template <typename ValueType>
class OpMap {
 public:
  /*!
   * \brief Check if the map has op as key.
   * \param op The key to the map
   * \return 1 if op is contained in map, 0 otherwise.
   */
  inline int count(const Op& op) const;
  /*!
   * \brief get the corresponding value element at op
   * \param op The key to the map
   * \return the const reference to the content value.
   */
  inline ValueType operator[](const Op& op) const;
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param op The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   */
  inline ValueType get(const Op& op, ValueType def_value) const;
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param expr The key to the map
   * \param def_value The default value when the key does not exist
   *         or if expr is not an Op.
   * \return the const reference to the content value.
   */
  inline ValueType get(const Expr& expr, ValueType def_value) const;

 private:
  friend class Op;
  // constructor
  explicit OpMap(const GenericOpMap& map) : map_(map) {}
  /*! \brief The internal map field */
  const GenericOpMap& map_;
};

// internal macros to make
#define RELAY_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::relay::OpRegistry& __make_##RelayOp

/*!
 * \def RELAY_REGISTER_OP
 * \brief Register a new operator, or set attribute of the corresponding op.
 *
 * \param OpName The name of registry
 *
 * \code
 *
 *  RELAY_REGISTER_OP("add")
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<OpKernel>("gpu_kernel", AddKernel);
 *
 * \endcode
 */
#define RELAY_REGISTER_OP(OpName)                        \
  DMLC_STR_CONCAT(RELAY_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::relay::OpRegistry::Registry()               \
          ->__REGISTER_OR_GET__(OpName)                  \
          .set_name()

// implementations
inline const OpNode* Op::operator->() const {
  return static_cast<const OpNode*>(node_.get());
}

template <typename ValueType>
inline OpMap<ValueType> Op::GetAttr(const std::string& key) {
  return OpMap<ValueType>(Op::GetGenericAttr(key));
}

inline OpNode* OpRegistry::get() {
  return const_cast<OpNode*>(op_.operator->());
}

inline OpRegistry& OpRegistry::describe(
    const std::string& descr) {  // NOLINT(*)
  get()->description = descr;
  return *this;
}

inline OpRegistry& OpRegistry::add_argument(const std::string& name,
                                            const std::string& type,
                                            const std::string& description) {
  auto n = make_node<AttrFieldInfoNode>();
  n->name = name;
  n->type_info = type;
  n->description = description;
  get()->arguments.push_back(AttrFieldInfo(n));
  return *this;
}

inline OpRegistry& OpRegistry::add_type_rel(
    const std::string& rel_name,
    runtime::TypedPackedFunc<bool(const Array<Type>&,
                                  int,
                                  const Attrs&,
                                  const TypeReporter&)> type_rel_func) {
  auto func_name = std::string("tvm.relay.type_relation.") + rel_name;
  TypeRelationFn env_type_rel_func;

  if (runtime::Registry::Get(func_name)) {
    auto env_func = EnvFunc::Get(func_name);
    env_type_rel_func = env_func;
  } else {
    runtime::Registry::Register(func_name)
        .set_body(type_rel_func.packed());
    auto env_func = EnvFunc::Get(func_name);
    env_type_rel_func = env_func;
  }

  Array<TypeVar> type_params;
  Array<Type> arg_types;

  // Add inputs.
  std::string input_name_prefix = "in";
  for (int i = 0; i < get()->num_inputs; i++) {
    auto name = input_name_prefix + std::to_string(i);
    auto param = TypeVarNode::make(name, Kind::kType);
    type_params.push_back(param);
    arg_types.push_back(param);
  }

  Array<Type> ty_call_args = arg_types;

  // Add output type.
  auto out_param = TypeVarNode::make("out", Kind::kType);
  type_params.push_back(out_param);
  // this will trigger copy on write.
  ty_call_args.push_back(out_param);

  // The attributes of primitive op is nullptr
  //
  // The attributes of primitive operator can vary at the call site.
  // The type of sum is also dependent on Attrs being passed.
  // So puting nullptr in the Attrs means that the operator is polymorphic on Attrs.
  //
  // A common example is sum(x, axis), where the choice of axis
  // can affect the type of the function.
  TypeConstraint type_rel =
      TypeRelationNode::make(env_type_rel_func,
                             ty_call_args,
                             arg_types.size(),
                             Attrs());

  auto func_type =
      FuncTypeNode::make(arg_types, out_param, type_params, {type_rel});

  get()->op_type = func_type;

  return *this;
}

inline OpRegistry& OpRegistry::set_num_inputs(int32_t n) {  // NOLINT(*)
  get()->num_inputs = n;
  return *this;
}

inline OpRegistry& OpRegistry::set_attrs_type_key(  // NOLINT(*)
    const std::string& type_key) {
  get()->attrs_type_key = type_key;
  get()->attrs_type_index = Node::TypeKey2Index(type_key.c_str());
  return *this;
}

inline OpRegistry& OpRegistry::set_support_level(int32_t n) {  // NOLINT(*)
  get()->support_level = n;
  return *this;
}

template <typename ValueType>
inline OpRegistry& OpRegistry::set_attr(  // NOLINT(*)
    const std::string& attr_name, const ValueType& value, int plevel) {
  CHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  TVMRetValue rv;
  rv = value;
  UpdateAttr(attr_name, rv, plevel);
  return *this;
}

// member functions of OpMap
inline int GenericOpMap::count(const Op& op) const {
  if (op.defined()) {
    const uint32_t idx = op->index_;
    return idx < data_.size() ? (data_[idx].second != 0) : 0;
  } else {
    return 0;
  }
}

inline const TVMRetValue& GenericOpMap::operator[](const Op& op) const {
  CHECK(op.defined());
  const uint32_t idx = op->index_;
  CHECK(idx < data_.size() && data_[idx].second != 0)
      << "Attribute " << attr_name_ << " has not been registered for Operator "
      << op->name;
  return data_[idx].first;
}

template <typename ValueType>
inline ValueType GenericOpMap::get(const Op& op, ValueType value) const {
  CHECK(op.defined());
  const uint32_t idx = op->index_;
  if (idx < data_.size() && data_[idx].second != 0) {
    return data_[idx].first;
  } else {
    return value;
  }
}

template <typename ValueType>
inline ValueType GenericOpMap::get(const Expr& expr, ValueType value) const {
  CHECK(expr.defined());
  if (const OpNode* op = expr.as<OpNode>()) {
    const uint32_t idx = op->index_;
    if (idx < data_.size() && data_[idx].second != 0) {
      return data_[idx].first;
    } else {
      return value;
    }
  } else {
    return value;
  }
}

template <typename ValueType>
inline int OpMap<ValueType>::count(const Op& op) const {
  return map_.count(op);
}

template <typename ValueType>
inline ValueType OpMap<ValueType>::operator[](const Op& op) const {
  return map_[op];
}

template <typename ValueType>
inline ValueType OpMap<ValueType>::get(const Op& op,
                                       ValueType def_value) const {
  return map_.get<ValueType>(op, def_value);
}

template <typename ValueType>
inline ValueType OpMap<ValueType>::get(const Expr& expr,
                                       ValueType def_value) const {
  return map_.get<ValueType>(expr, def_value);
}


/*!
 * \brief Check that an expression is a "primtive operator".
 *
 * Will return true if the expression is an operator which
 * matches the form of primtive operators registered directly
 * by the Relay codebase.
 *
 * That is the arguments are all type variables, and there is a single
 * type relation applied to the input and output types.
 */
inline bool IsPrimitiveOp(const Expr& expr) {
  const auto* op = expr.as<OpNode>();
  return op != nullptr && op->IsPrimitiveOp();
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_H_
