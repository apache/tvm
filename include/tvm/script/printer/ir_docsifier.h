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
#ifndef TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
#define TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_

#include <tvm/node/node.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/frame.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>
#include <tvm/script/printer/var_table.h>
#include <tvm/support/with.h>

namespace tvm {
namespace script {
namespace printer {

using WithCtx = With<ContextManager>;

/*!
 * \brief IRDocsifier is the top-level interface in the IR->Doc process.
 *
 * It provides methods to convert IR node object to Doc, operate on Frame
 * objects and change dispatch tokens.
 *
 * Example usage:
 * \code
 * TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
 *    .set_dispatch([](TracedObject<tir::Var> obj, IRDocsifier p) { return IdDoc("x"); });
 *
 * TracedObject<tir::Var> var = ...;
 * IRDocsifier p;
 * p->AsDoc(var); // returns an IdDoc("x")
 * \endcode
 *
 */
class IRDocsifierNode : public Object {
 public:
  /*!
   * \brief The var table to use during the printing process.
   * \sa VarTableNode
   */
  VarTable vars;
  /*!
   * \brief The stack of frames.
   * \sa FrameNode
   */
  Array<Frame> frames;
  /*!
   * \brief The stack of dispatch tokens.
   *
   * The dispatch token on the top decides which dispatch function to use
   * when converting IR node object to Doc.
   */
  Array<String> dispatch_tokens;
  /*!
   * \brief This map connects IR dipatch token to the name of identifier.
   */
  Map<String, String> ir_prefix;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("vars", &vars);
    v->Visit("frames", &frames);
    v->Visit("dispatch_tokens", &dispatch_tokens);
    v->Visit("ir_prefix", &ir_prefix);
  }

  static constexpr const char* _type_key = "script.printer.IRDocsifier";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRDocsifierNode, Object);

 public:
  /*!
   * \brief Transform the input object into TDoc.
   * \param obj The object to be transformed.
   *
   * \return The Doc object.
   */
  template <class TDoc>
  TDoc AsDoc(const TracedObject<ObjectRef>& obj) const {
    auto result = Downcast<TDoc>(AsDocImpl(obj));
    result->source_paths.push_back(obj.GetPath());
    return result;
  }

  /*!
   * \brief Helper method to transform object into ExprDoc.
   * \param obj The object to be transformed.
   *
   * \return The ExprDoc object.
   */
  ExprDoc AsExprDoc(const TracedObject<ObjectRef>& obj) { return AsDoc<ExprDoc>(obj); }

  /*!
   * \brief Push a new dispatch token into the stack
   * \details The top dispatch token decides which dispatch table to use
   *          when printing Object. This method returns a RAII guard which
   *          pops the token when going out of the scope.
   *
   * \param token The dispatch token to push.
   *
   * \return A RAII guard to pop dispatch token when going out of scope.
   */
  WithCtx WithDispatchToken(const String& token) {
    this->dispatch_tokens.push_back(token);
    return WithCtx(nullptr, [this]() { this->dispatch_tokens.pop_back(); });
  }

  /*!
   * \brief Push a new frame the stack
   * \details Frame contains the contextual information that's needed during printing,
   *          for example, variables in the scope. This method returns a RAII guard which
   *          pops the frame and call the cleanup method of frame when going out of the scope.
   *
   * \param frame The frame to push.
   *
   * \return A RAII guard to pop frame and call the exit method of frame
   *          when going out of scope
   */
  WithCtx WithFrame(const Frame& frame) {
    frame->EnterWithScope();
    this->frames.push_back(frame);
    return WithCtx(nullptr, [this, pushed_frame = frame]() {
      Frame last_frame = this->frames.back();
      ICHECK_EQ(last_frame, pushed_frame);
      this->frames.pop_back();
      last_frame->ExitWithScope();
    });
  }

  /*!
   * \brief Get the top frame with type FrameType
   * \tparam FrameType The type of frame to get.
   */
  template <typename FrameType>
  Optional<FrameType> GetFrame() const {
    for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
      if (const auto* f = (*it).as<typename FrameType::ContainerType>()) {
        return GetRef<FrameType>(f);
      }
    }
    return NullOpt;
  }

 private:
  Doc AsDocImpl(const TracedObject<ObjectRef>& obj) const;
};

/*!
 * \brief Reference type of IRDocsifierNode.
 */
class IRDocsifier : public ObjectRef {
 public:
  /*!
   * \brief Create a IRDocsifier.
   * \param ir_prefix The ir_prefix to use for this IRDocsifier.
   */
  explicit IRDocsifier(Map<String, String> ir_prefix);

  using FType = TracedObjectFunctor<printer::Doc, IRDocsifier>;
  /*!
   * \brief The registration table for IRDocsifier.
   */
  TVM_DLL static FType& vtable();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRDocsifier, ObjectRef, IRDocsifierNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
