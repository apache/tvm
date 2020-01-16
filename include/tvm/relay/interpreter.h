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
#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relay {

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
runtime::TypedPackedFunc<ObjectRef(Expr)>
CreateInterpreter(IRModule mod, DLContext context, Target target);

/*! \brief A Relay closure, i.e a scope and a function. */
class Closure;

/*! \brief The container type of Closures. */
class ClosureNode : public Object {
 public:
  /*! \brief The set of free variables in the closure.
   *
   * These are the captured variables which are required for
   * evaluation when we call the closure.
   */
  tvm::Map<Var, ObjectRef> env;
  /*! \brief The function which implements the closure.
   *
   * \note May reference the variables contained in the env.
   */
  Function func;

  ClosureNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("env", &env);
    v->Visit("func", &func);
  }

  TVM_DLL static Closure make(tvm::Map<Var, ObjectRef> env, Function func);

  static constexpr const char* _type_key = "relay.Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(ClosureNode, Object);
};

class Closure : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Closure, ObjectRef, ClosureNode);
};

/*! \brief A Relay Recursive Closure. A closure that has a name. */
class RecClosure;

/*! \brief The container type of RecClosure. */
class RecClosureNode : public Object {
 public:
  /*! \brief The closure. */
  Closure clos;
  /*! \brief variable the closure bind to. */
  Var bind;

  RecClosureNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("clos", &clos);
    v->Visit("bind", &bind);
  }

  TVM_DLL static RecClosure make(Closure clos, Var bind);

  static constexpr const char* _type_key = "relay.RecClosure";
  TVM_DECLARE_FINAL_OBJECT_INFO(RecClosureNode, Object);
};

class RecClosure : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RecClosure, ObjectRef, RecClosureNode);
};

/*! \brief A tuple value. */
class TupleValue;

/*! \brief Tuple (x, ... y). */
struct TupleValueNode : Object {
  tvm::Array<ObjectRef> fields;

  TupleValueNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("fields", &fields); }

  TVM_DLL static TupleValue make(tvm::Array<ObjectRef> value);

  static constexpr const char* _type_key = "relay.TupleValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleValueNode, Object);
};

class TupleValue : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TupleValue, ObjectRef, TupleValueNode);
};

/*! \brief A reference value. */
class RefValue;

struct RefValueNode : Object {
  mutable ObjectRef value;

  RefValueNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
  }

  TVM_DLL static RefValue make(ObjectRef val);

  static constexpr const char* _type_key = "relay.RefValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefValueNode, Object);
};

class RefValue : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RefValue, ObjectRef, RefValueNode);
};

/*! \brief An ADT constructor value. */
class ConstructorValue;

struct ConstructorValueNode : Object {
  int32_t tag;

  tvm::Array<ObjectRef> fields;

  /*! \brief Optional field tracking ADT constructor. */
  Constructor constructor;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tag", &tag);
    v->Visit("fields", &fields);
    v->Visit("constructor", &constructor);
  }

  TVM_DLL static ConstructorValue make(int32_t tag,
                                       tvm::Array<ObjectRef> fields,
                                       Constructor construtor = {});

  static constexpr const char* _type_key = "relay.ConstructorValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstructorValueNode, Object);
};

class ConstructorValue : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ConstructorValue, ObjectRef, ConstructorValueNode);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_INTERPRETER_H_
