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

#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/object.h>
#include <tvm/target/target.h>

#include <unordered_set>

namespace tvm {
namespace relay {

/*! \brief The container type of Closures used by the interpreter. */
class InterpreterClosureObj : public runtime::ClosureObj {
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

  InterpreterClosureObj() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("env", &env);
    v->Visit("func", &func);
  }

  static constexpr const char* _type_key = "interpreter.Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(InterpreterClosureObj, runtime::ClosureObj);
};

class InterpreterClosure : public runtime::Closure {
 public:
  TVM_DLL InterpreterClosure(tvm::Map<Var, ObjectRef> env, Function func);
  TVM_DEFINE_OBJECT_REF_METHODS(InterpreterClosure, runtime::Closure, InterpreterClosureObj);
};

/*! \brief The container type of RecClosure. */
class RecClosureObj : public Object {
 public:
  /*! \brief The closure. */
  InterpreterClosure clos;
  /*! \brief variable the closure bind to. */
  Var bind;

  RecClosureObj() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("clos", &clos);
    v->Visit("bind", &bind);
  }

  static constexpr const char* _type_key = "interpreter.RecClosure";
  TVM_DECLARE_FINAL_OBJECT_INFO(RecClosureObj, Object);
};

class RecClosure : public ObjectRef {
 public:
  TVM_DLL RecClosure(InterpreterClosure clos, Var bind);
  TVM_DEFINE_OBJECT_REF_METHODS(RecClosure, ObjectRef, RecClosureObj);
};

struct RefValueObj : Object {
  mutable ObjectRef value;

  RefValueObj() {}

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("value", &value); }

  static constexpr const char* _type_key = "relay.RefValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefValueObj, Object);
};

class RefValue : public ObjectRef {
 public:
  TVM_DLL RefValue(ObjectRef val);
  TVM_DEFINE_OBJECT_REF_METHODS(RefValue, ObjectRef, RefValueObj);
};

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

  static constexpr const char* _type_key = "relay.ConstructorValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstructorValueObj, Object);
};

class ConstructorValue : public ObjectRef {
 public:
  TVM_DLL ConstructorValue(int32_t tag, tvm::Array<ObjectRef> fields, Constructor construtor = {});

  TVM_DEFINE_OBJECT_REF_METHODS(ConstructorValue, ObjectRef, ConstructorValueObj);
};

/*!
 * \brief Returns a packed function over Relay expressions which will evaluate \p expr
 * applied to those arguments, where \p expr is w.r.t. the definitions in \p mod.
 *
 * This function is intended to support the Python 'debug' executor.
 *
 * The given \p expr should have function type. The given \p mod may be empty or
 * undefined if \p expr is self-contained. Relay arguments passed to the result
 * packed function must be constants, references, or constructors/tuples over such.
 * As much work as possible is done while constructing the result packed function, and
 * that function may be reasonably efficiently applied multiple times without redoing
 * unnecessary work.
 *
 * Primitives are lowered and compiled to packed functions for execution on \p device
 * with properties given by \p target. All other Relay constructs are interpreted.
 *
 * The interpreter is intended to be a 'reference' implementation of the Relay semantics
 * for testing and interactive use. It is not intended to be particularly efficient.
 *
 * \param mod A module containing definitions which can be referenced from
 * \p expr. May be empty or undefined.
 * \param expr An expression of function type to evaluate. May reference definitions from \p mod.
 * \param device The device on which all primitives will be executed.
 * \param target The compiler target flag for compiling primitives.
 * \return A packed function that takes an array of Relay expressions and returns the
 * result of applying \p expr to those arguments.
 */
TypedPackedFunc<ObjectRef(Array<Expr>)> EvalFunction(IRModule mod, Expr expr, Device device,
                                                     Target target);

/*!
 * \brief Evaluates \p expr and returns its result.
 *
 * This function is intended to support TVM constant evaluation.
 *
 * \param expr An expression to evaluate.
 * \param type_definitions Global type definitions which \p expr may references.
 * \param import_set Already imported external modules.
 * \param device The device on which all primitives will be executed.
 * \param target The compiler target flag for compiling primitives.
 * \param attrs Attributes for the expression to be evaluated with
 * @return The object representing the result.
 */
ObjectRef Eval(Expr expr, Map<GlobalTypeVar, TypeData> type_definitions,
               std::unordered_set<String> import_set, Device device, Target target,
               Map<String, ObjectRef> attrs = {});

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_INTERPRETER_H_
