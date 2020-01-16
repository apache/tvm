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
 * can be produced by a Relay program and are exposed via TVM's object
 * protocol to Python for introspection and debugging.
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
#include <tvm/runtime/container.h>
#include <tvm/runtime/vm.h>

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

/*! \brief A Relay Recursive Closure. A closure that has a name. */
class RecClosure;

/*! \brief The container type of RecClosure. */
class RecClosureObj : public Object {
 public:
  /*! \brief The closure. */
  runtime::vm::Closure clos;
  /*! \brief variable the closure bind to. */
  Var bind;

  RecClosureObj() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("clos", &clos);
    v->Visit("bind", &bind);
  }

  TVM_DLL static RecClosure make(runtime::vm::Closure clos, Var bind);

  static constexpr const char* _type_key = "relay.RecClosure";
  TVM_DECLARE_FINAL_OBJECT_INFO(RecClosureObj, Object);
};

class RecClosure : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RecClosure, ObjectRef, RecClosureObj);
};

/*! \brief A reference value. */
class RefValue;

struct RefValueObj : Object {
  mutable ObjectRef value;

  RefValueObj() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
  }

  TVM_DLL static RefValue make(ObjectRef val);

  static constexpr const char* _type_key = "relay.RefValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefValueObj, Object);
};

class RefValue : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RefValue, ObjectRef, RefValueObj);
};

/*! \brief An ADT constructor value. */
class ConstructorValue;

struct ConstructorValueObj : Object {
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
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstructorValueObj, Object);
};

class ConstructorValue : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ConstructorValue, ObjectRef, ConstructorValueObj);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_INTERPRETER_H_
