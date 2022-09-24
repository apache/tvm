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
 * \file tvm/ir/op.h
 * \brief Primitive operators(builtin intrinsics)
 *        and registry for them.
 */
#ifndef TVM_IR_OP_H_
#define TVM_IR_OP_H_

#include <dmlc/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/node/attr_registry_map.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {

// forward declare name.
template <typename>
class OpAttrMap;

// TODO(tvm-team): migrate low-level intrinsics to use Op
/*!
 * \brief Primitive Op(builtin intrinsics)
 *
 * This data structure stores the meta-data
 * about primitive operators that can be invoked via Call.
 *
 * Low-level IR intrinsics(such as libc.expf) are also
 * implemented via Op.
 *
 * \sa Op
 */
class OpNode : public RelayExprNode {
 public:
  /*! \brief name of the operator */
  String name;
  /*! \brief the type of the operator */
  mutable FuncType op_type;
  /*!
   * \brief detailed description of the operator
   *  This can be used to generate docstring automatically for the operator.
   */
  String description;
  /* \brief Information of input arguments to the operator */
  Array<AttrFieldInfo> arguments;
  /*!
   * \brief The type key of the attribute field
   *  This can be empty, in which case it defaults to anything.
   */
  String attrs_type_key;
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("op_type", &op_type);
    v->Visit("description", &description);
    v->Visit("arguments", &arguments);
    v->Visit("attrs_type_key", &attrs_type_key);
    v->Visit("num_inputs", &num_inputs);
    v->Visit("support_level", &support_level);
  }

  bool SEqualReduce(const OpNode* other, SEqualReducer equal) const {
    // pointer equality is fine as there is only one op with the same name.
    return this == other;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    // Name uniquely identifies an Op.
    hash_reduce(name);
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

  static constexpr const char* _type_key = "Op";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpNode, RelayExprNode);

 private:
  /*! \return the internal attr registry index. */
  uint32_t AttrRegistryIndex() const { return index_; }
  /*! \brief repr to be printed in registry*/
  std::string AttrRegistryName() const { return name; }

  // friend class
  template <typename>
  friend class AttrRegistryMapContainerMap;
  template <typename, typename>
  friend class AttrRegistry;
  friend class OpRegEntry;

  friend bool IsPrimitiveOp(const RelayExpr&);
  // Program internal unique index of operator.
  // Used to help index the program.
  uint32_t index_{0};
  // whether this is a primitive op. -1 means unknown.
  mutable int is_primitive_{-1};
  // Internal function to compute if it is primitive op
  bool IsPrimitiveOp_() const {
    const auto& fn_ty = this->op_type;
    ICHECK(fn_ty.get() != nullptr) << "op_type of " << this->name << " is not registered";
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
 * \brief Managed reference class to OpNode.
 * \sa OpNode
 */
class Op : public RelayExpr {
 public:
  /*!
   * \brief Get additional registered attribute about operators.
   *  If nothing has been registered, an empty OpAttrMap will be returned.
   * \param attr_name The name of the attribute.
   * \return An OpAttrMap of specified attr_name.
   * \tparam ValueType The type of the attribute.
   */
  template <typename ValueType>
  inline static OpAttrMap<ValueType> GetAttrMap(const String& attr_name);
  /*!
   * \brief Checks if an attr map is present in the registry.
   * \param attr_name The name of the attribute.
   * \return bool True if the attr is present.
   */
  TVM_DLL static bool HasAttrMap(const String& attr_name);
  /*!
   * \brief Get an Op for a given operator name.
   *  Will raise an error if the op has not been registered.
   * \param op_name Name of the operator.
   * \return Pointer to a Op, valid throughout program lifetime.
   */
  TVM_DLL static const Op& Get(const String& op_name);

  TVM_DEFINE_OBJECT_REF_METHODS(Op, RelayExpr, OpNode)

 private:
  /*!
   * \brief Get generic attrmap given attr name
   * \param key The attribute key
   * \return The attr map.
   */
  TVM_DLL static const AttrRegistryMapContainerMap<Op>& GetAttrMapContainer(const String& key);
};

/*!
 * \brief Helper structure to register operators
 * \sa TVM_REGISTER_OP
 */
class OpRegEntry {
 public:
  /*! \return the operator */
  const Op& op() const { return op_; }
  /*!
   * \brief setter function during registration
   *  Set the description of operator
   * \param descr the description string.
   * \return reference to self.
   */
  inline OpRegEntry& describe(const std::string& descr);  // NOLINT(*)
  /*!
   * \brief Add argument information to the function.
   * \param name Name of the argument.
   * \param type Type of the argument.
   * \param description Description of the argument.
   * \return reference to self.
   */
  inline OpRegEntry& add_argument(const std::string& name, const std::string& type,
                                  const std::string& description);
  /*!
   * \brief Attach the type function corresponding to the return type.
   * \param rel_name The type relation name to register.
   * \param type_rel_func The backing relation function which can solve an arbitrary
   * relation on variables.
   * \return reference to self.
   */
  inline OpRegEntry& add_type_rel(
      const std::string& rel_name,
      runtime::TypedPackedFunc<bool(const Array<Type>&, int, const Attrs&, const TypeReporter&)>
          type_rel_func);
  /*!
   * \brief Set the attrs type key and index to be AttrsType.
   * \tparam AttrsType the attribute type to b set.
   * \return reference to self.
   */
  template <typename AttrsType>
  inline OpRegEntry& set_attrs_type();
  /*!
   * \brief Set the attrs type key and index to be AttrsType.
   * \param key The attribute type key to be set.
   * \return reference to self.
   */
  inline OpRegEntry& set_attrs_type_key(const String& key);
  /*!
   * \brief Set the num_inputs
   * \param n The number of inputs to be set.
   * \return reference to self.
   */
  inline OpRegEntry& set_num_inputs(int32_t n);  // NOLINT(*)
  /*!
   * \brief Set the support level of op.
   * \param level The support level.
   * \return reference to self.
   */
  inline OpRegEntry& set_support_level(int32_t level);  // NOLINT(*)
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
  inline OpRegEntry& set_attr(const std::string& attr_name,  // NOLINT(*)
                              const ValueType& value, int plevel = 10);

  /*!
   * \brief Resets an attr of the registry.
   * \param attr_name The name of the attribute.
   */
  inline void reset_attr(const std::string& attr_name);

  // set the name of the op to be the same as registry
  inline OpRegEntry& set_name() {  // NOLINT(*)
    if (get()->name.length() == 0) {
      get()->name = name;
    }
    return *this;
  }
  /*!
   * \brief Register or get a new entry.
   * \param name The name of the operator.
   * \return the corresponding entry.
   */
  TVM_DLL static OpRegEntry& RegisterOrGet(const String& name);

 private:
  template <typename, typename>
  friend class AttrRegistry;
  // the name
  std::string name;
  /*! \brief The operator */
  Op op_;
  // private constructor
  TVM_DLL OpRegEntry(uint32_t reg_index);
  // return internal pointer to op.
  inline OpNode* get();
  // update the attribute OpAttrMap
  TVM_DLL void UpdateAttr(const String& key, runtime::TVMRetValue value, int plevel);
};

/*!
 * \brief Map<Op,ValueType> used to store meta-information about Op.
 * \tparam ValueType The type of the value stored in map.
 */
template <typename ValueType>
class OpAttrMap : public AttrRegistryMap<Op, ValueType> {
 public:
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param expr The key to the map
   * \param def_value The default value when the key does not exist
   *         or if expr is not an Op.
   * \return the const reference to the content value.
   */
  inline ValueType get(const RelayExpr& expr, ValueType def_value) const;

  using TParent = AttrRegistryMap<Op, ValueType>;
  using TParent::count;
  using TParent::get;
  using TParent::operator[];

 private:
  friend class Op;
  // constructor
  explicit OpAttrMap(const AttrRegistryMapContainerMap<Op>& map) : TParent(map) {}
};

// internal macros to make
#define TVM_OP_REGISTER_VAR_DEF static DMLC_ATTRIBUTE_UNUSED ::tvm::OpRegEntry& __make_##Op

/*!
 * \def TVM_REGISTER_OP
 * \brief Register a new operator, or set attribute of the corresponding op.
 *
 * \param OpName The name of registry
 *
 * \code
 *
 *  TVM_REGISTER_OP("add")
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<OpKernel>("gpu_kernel", AddKernel);
 *
 * \endcode
 */
#define TVM_REGISTER_OP(OpName)                          \
  TVM_STR_CONCAT(TVM_OP_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::OpRegEntry::RegisterOrGet(OpName).set_name()

// implementations

template <typename ValueType>
inline OpAttrMap<ValueType> Op::GetAttrMap(const String& key) {
  return OpAttrMap<ValueType>(Op::GetAttrMapContainer(key));
}

inline OpNode* OpRegEntry::get() { return const_cast<OpNode*>(op_.operator->()); }

inline OpRegEntry& OpRegEntry::describe(const std::string& descr) {  // NOLINT(*)
  get()->description = descr;
  return *this;
}

inline OpRegEntry& OpRegEntry::add_argument(const std::string& name, const std::string& type,
                                            const std::string& description) {
  auto n = make_object<AttrFieldInfoNode>();
  n->name = name;
  n->type_info = type;
  n->description = description;
  get()->arguments.push_back(AttrFieldInfo(n));
  return *this;
}

inline OpRegEntry& OpRegEntry::add_type_rel(
    const std::string& rel_name,
    runtime::TypedPackedFunc<bool(const Array<Type>&, int, const Attrs&, const TypeReporter&)>
        type_rel_func) {
  auto func_name = std::string("tvm.relay.type_relation.") + rel_name;
  TypeRelationFn env_type_rel_func;

  if (runtime::Registry::Get(func_name)) {
    auto env_func = EnvFunc::Get(func_name);
    env_type_rel_func = env_func;
  } else {
    runtime::Registry::Register(func_name).set_body(type_rel_func.packed());
    auto env_func = EnvFunc::Get(func_name);
    env_type_rel_func = env_func;
  }

  Array<TypeVar> type_params;
  Array<Type> arg_types;

  // Add inputs.
  std::string input_name_prefix = "in";
  for (int i = 0; i < get()->num_inputs; i++) {
    auto name = input_name_prefix + std::to_string(i);
    auto param = TypeVar(name, TypeKind::kType);
    type_params.push_back(param);
    arg_types.push_back(param);
  }

  Array<Type> ty_call_args = arg_types;

  // Add output type.
  auto out_param = TypeVar("out", TypeKind::kType);
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
      TypeRelation(env_type_rel_func, ty_call_args, arg_types.size(), Attrs());

  auto func_type = FuncType(arg_types, out_param, type_params, {type_rel});

  get()->op_type = func_type;

  return *this;
}

inline OpRegEntry& OpRegEntry::set_num_inputs(int32_t n) {  // NOLINT(*)
  get()->num_inputs = n;
  return *this;
}

template <typename AttrsType>
inline OpRegEntry& OpRegEntry::set_attrs_type() {  // NOLINT(*)
  get()->attrs_type_key = AttrsType::_type_key;
  get()->attrs_type_index = AttrsType::RuntimeTypeIndex();
  return *this;
}

inline OpRegEntry& OpRegEntry::set_attrs_type_key(const String& key) {  // NOLINT(*)
  get()->attrs_type_key = key;
  get()->attrs_type_index = Object::TypeKey2Index(key);
  return *this;
}

inline OpRegEntry& OpRegEntry::set_support_level(int32_t n) {  // NOLINT(*)
  get()->support_level = n;
  return *this;
}

template <typename ValueType>
inline OpRegEntry& OpRegEntry::set_attr(  // NOLINT(*)
    const std::string& attr_name, const ValueType& value, int plevel) {
  ICHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  runtime::TVMRetValue rv;
  rv = value;
  UpdateAttr(attr_name, rv, plevel);
  return *this;
}

// member functions of OpAttrMap

template <typename ValueType>
inline ValueType OpAttrMap<ValueType>::get(const RelayExpr& expr, ValueType def_value) const {
  ICHECK(expr.defined());
  if (const OpNode* op = expr.as<OpNode>()) {
    return this->map_.get(GetRef<Op>(op), def_value);
  } else {
    return def_value;
  }
}

/*!
 * \brief Check that an expression is a "primitive operator".
 *
 * Will return true if the expression is an operator which
 * matches the form of primitive operators registered directly
 * by the Relay codebase.
 *
 * That is the arguments are all type variables, and there is a single
 * type relation applied to the input and output types.
 *
 * \param expr An expression.
 * \return Whether the expression is primitive op.
 */
inline bool IsPrimitiveOp(const RelayExpr& expr) {
  const auto* op = expr.as<OpNode>();
  return op != nullptr && op->IsPrimitiveOp();
}

}  // namespace tvm
#endif  // TVM_IR_OP_H_
