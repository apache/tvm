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
 * \file tvm/relay/interpreter.h
 * \brief An interpreter for Relay.
 *
 * This file implements a simple reference interpreter for Relay programs.
 * Given a Relay module, and a Relay expression it produces a value.
 *
 * The interpreter's values are a naive representation of the values that
 * can be produced by a Relay program and are exposed via tvm::Node's
 * system to Python for introspection and debugging.
 *
 * The interpreter's intent is to serve as a reference semantics for the Relay IR,
 * as well as for debugging and testing.
 */
#ifndef TVM_RELAY_INTERPRETER_H_
#define TVM_RELAY_INTERPRETER_H_

#include <tvm/build_module.h>
#include <tvm/relay/module.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

/*!
 * \brief A Relay value.
 */
class Value;

/*!
 *\brief Create a Interpreter function that can
 *  evaluate an expression and produce a value.
 *
 * The resulting value can be passed to Python, making it easy to use
 * for testing and debugging.
 *
 * The interpreter interprets the program fragments not supported by the
 * TVM runtime, although the interpreter is naively implemented it uses
 * TVM operators for evaluating all operators.
 *
 * Our intent is that this will never be the most efficient implementation of
 * Relay's semantics, but a readable and clear one.
 *
 * \param mod The function module.
 * \param context The primary context that the interepreter runs on.
 * \param target Compiler target flag to compile the functions on the context.
 * \return A function that takes in an expression and returns a value.
 */
runtime::TypedPackedFunc<Value(Expr)>
CreateInterpreter(Module mod, DLContext context, Target target);

/*! \brief The base container type of Relay values. */
class ValueNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Value";
  TVM_DECLARE_BASE_NODE_INFO(ValueNode, RelayNode);
};

class Value : public NodeRef {
 public:
  Value() {}
  explicit Value(NodePtr<Node> n) : NodeRef(n) {}
  const ValueNode* operator->() const {
    return static_cast<const ValueNode*>(node_.get());
  }

  using ContainerType = ValueNode;
};

/*! \brief A Relay closure, i.e a scope and a function. */
class Closure;

/*! \brief The container type of Closures. */
class ClosureNode : public ValueNode {
 public:
  /*! \brief The set of free variables in the closure.
   *
   * These are the captured variables which are required for
   * evaluation when we call the closure.
   */
  tvm::Map<Var, Value> env;
  /*! \brief The function which implements the closure.
   *
   * \note May reference the variables contained in the env.
   */
  Function func;

  ClosureNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("env", &env);
    v->Visit("func", &func);
  }

  TVM_DLL static Closure make(tvm::Map<Var, Value> env, Function func);

  static constexpr const char* _type_key = "relay.Closure";
  TVM_DECLARE_NODE_TYPE_INFO(ClosureNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(Closure, ClosureNode, Value);

/*! \brief A tuple value. */
class TupleValue;

/*! \brief Tuple (x, ... y). */
struct TupleValueNode : ValueNode {
  tvm::Array<Value> fields;

  TupleValueNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final { v->Visit("fields", &fields); }

  TVM_DLL static TupleValue make(tvm::Array<Value> value);

  static constexpr const char* _type_key = "relay.TupleValue";
  TVM_DECLARE_NODE_TYPE_INFO(TupleValueNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(TupleValue, TupleValueNode, Value);

/*! \brief A tensor value. */
class TensorValue;

/*! \brief The tensor value container, wrapping an NDArray. */
struct TensorValueNode : ValueNode {
  runtime::NDArray data;

  TensorValueNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final { v->Visit("data", &data); }

  /*! \brief Build a value from an NDArray. */
  TVM_DLL static TensorValue make(runtime::NDArray data);

  static constexpr const char* _type_key = "relay.TensorValue";
  TVM_DECLARE_NODE_TYPE_INFO(TensorValueNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(TensorValue, TensorValueNode, Value);

/*! \brief A reference value. */
class RefValue;

struct RefValueNode : ValueNode {
  mutable Value value;

  RefValueNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("value", &value);
  }

  TVM_DLL static RefValue make(Value val);

  static constexpr const char* _type_key = "relay.RefValue";
  TVM_DECLARE_NODE_TYPE_INFO(RefValueNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(RefValue, RefValueNode, Value);

/*! \brief An ADT constructor value. */
class ConstructorValue;

struct ConstructorValueNode : ValueNode {
  int32_t tag;

  tvm::Array<Value> fields;

  /*! \brief Optional field tracking ADT constructor. */
  Constructor constructor;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("tag", &tag);
    v->Visit("fields", &fields);
    v->Visit("constructor", &constructor);
  }

  TVM_DLL static ConstructorValue make(int32_t tag,
                                       tvm::Array<Value> fields,
                                       Constructor construtor = {});

  static constexpr const char* _type_key = "relay.ConstructorValue";
  TVM_DECLARE_NODE_TYPE_INFO(ConstructorValueNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(ConstructorValue, ConstructorValueNode, Value);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_INTERPRETER_H_
