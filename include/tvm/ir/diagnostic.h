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
 * \file diagnostic.h
 * \brief A new diagnostic interface for TVM error reporting.
 *
 */

#ifndef TVM_IR_DIAGNOSTIC_H_
#define TVM_IR_DIAGNOSTIC_H_

#include <tvm/ir/module.h>

#include <sstream>
#include <string>

namespace tvm {

using tvm::runtime::TypedPackedFunc;

/*! \brief The diagnostic level, controls the printing of the message. */
enum class DiagnosticLevel : int {
  kBug = 10,
  kError = 20,
  kWarning = 30,
  kNote = 40,
  kHelp = 50,
};

class DiagnosticBuilder;

/*! \brief A compiler diagnostic. */
class Diagnostic;

/*! \brief A compiler diagnostic message. */
class DiagnosticNode : public Object {
 public:
  /*! \brief The level. */
  DiagnosticLevel level;
  /*! \brief The span at which to report an error. */
  Span span;
  /*!
   * \brief The object location at which to report an error.
   *
   * The object loc provides a location when span is not always
   * available during transformation. The error reporter can
   * still pick up loc->span if necessary.
   */
  ObjectRef loc;
  /*! \brief The diagnostic message. */
  String message;

  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("level", &level);
    v->Visit("span", &span);
    v->Visit("message", &message);
  }

  bool SEqualReduce(const DiagnosticNode* other, SEqualReducer equal) const {
    return equal(this->level, other->level) && equal(this->span, other->span) &&
           equal(this->message, other->message);
  }

  static constexpr const char* _type_key = "Diagnostic";
  TVM_DECLARE_FINAL_OBJECT_INFO(DiagnosticNode, Object);
};

class Diagnostic : public ObjectRef {
 public:
  TVM_DLL Diagnostic(DiagnosticLevel level, Span span, const std::string& message);

  static DiagnosticBuilder Bug(Span span);
  static DiagnosticBuilder Error(Span span);
  static DiagnosticBuilder Warning(Span span);
  static DiagnosticBuilder Note(Span span);
  static DiagnosticBuilder Help(Span span);
  // variants uses object location
  static DiagnosticBuilder Bug(ObjectRef loc);
  static DiagnosticBuilder Error(ObjectRef loc);
  static DiagnosticBuilder Warning(ObjectRef loc);
  static DiagnosticBuilder Note(ObjectRef loc);
  static DiagnosticBuilder Help(ObjectRef loc);
  // variants uses object ptr.
  static DiagnosticBuilder Bug(const Object* loc);
  static DiagnosticBuilder Error(const Object* loc);
  static DiagnosticBuilder Warning(const Object* loc);
  static DiagnosticBuilder Note(const Object* loc);
  static DiagnosticBuilder Help(const Object* loc);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Diagnostic, ObjectRef, DiagnosticNode);
};

/*!
 * \brief A wrapper around std::stringstream to build a diagnostic.
 */
class DiagnosticBuilder {
 public:
  /*! \brief The level. */
  DiagnosticLevel level;

  /*! \brief The source name. */
  SourceName source_name;

  /*! \brief The span of the diagnostic. */
  Span span;

  /*!
   * \brief The object location at which to report an error.
   */
  ObjectRef loc;

  template <typename T>
  DiagnosticBuilder& operator<<(const T& val) {  // NOLINT(*)
    stream_ << val;
    return *this;
  }

  DiagnosticBuilder() : level(DiagnosticLevel::kError), source_name(), span(Span()) {}

  DiagnosticBuilder(const DiagnosticBuilder& builder)
      : level(builder.level), source_name(builder.source_name), span(builder.span) {}

  DiagnosticBuilder(DiagnosticLevel level, Span span) : level(level), span(span) {}

  DiagnosticBuilder(DiagnosticLevel level, ObjectRef loc) : level(level), loc(loc) {}

  operator Diagnostic() { return Diagnostic(this->level, this->span, this->stream_.str()); }

 private:
  std::stringstream stream_;
  friend class Diagnostic;
};

/*!
 * \brief A diagnostic context for recording errors against a source file.
 */
class DiagnosticContext;

/*! \brief Display diagnostics in a given display format.
 *
 * A diagnostic renderer is responsible for converting the
 * raw diagnostics into consumable output.
 *
 * For example the terminal renderer will render a sequence
 * of compiler diagnostics to std::out and std::err in
 * a human readable form.
 */
class DiagnosticRendererNode : public Object {
 public:
  TypedPackedFunc<void(DiagnosticContext ctx)> renderer;

  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "DiagnosticRenderer";
  TVM_DECLARE_FINAL_OBJECT_INFO(DiagnosticRendererNode, Object);
};

class DiagnosticRenderer : public ObjectRef {
 public:
  TVM_DLL DiagnosticRenderer(TypedPackedFunc<void(DiagnosticContext ctx)> render);
  TVM_DLL DiagnosticRenderer()
      : DiagnosticRenderer(TypedPackedFunc<void(DiagnosticContext ctx)>()) {}

  void Render(const DiagnosticContext& ctx);

  DiagnosticRendererNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<DiagnosticRendererNode*>(get_mutable());
  }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DiagnosticRenderer, ObjectRef, DiagnosticRendererNode);
};

class DiagnosticContextNode : public Object {
 public:
  /*! \brief The Module to report against. */
  IRModule module;

  /*! \brief The set of diagnostics to report. */
  Array<Diagnostic> diagnostics;

  /*! \brief The renderer set for the context. */
  DiagnosticRenderer renderer;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("module", &module);
    v->Visit("diagnostics", &diagnostics);
  }

  bool SEqualReduce(const DiagnosticContextNode* other, SEqualReducer equal) const {
    return equal(module, other->module) && equal(diagnostics, other->diagnostics);
  }

  static constexpr const char* _type_key = "DiagnosticContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(DiagnosticContextNode, Object);
};

class DiagnosticContext : public ObjectRef {
 public:
  TVM_DLL DiagnosticContext(const IRModule& module, const DiagnosticRenderer& renderer);
  TVM_DLL static DiagnosticContext Default(const IRModule& source_map);

  /*! \brief Emit a diagnostic.
   * \param diagnostic The diagnostic to emit.
   */
  void Emit(const Diagnostic& diagnostic);

  /*! \brief Emit a diagnostic and then immediately attempt to render all errors.
   *
   * \param diagnostic The diagnostic to emit.
   *
   * Note: this will raise an exception if you would like to instead continue execution
   * use the Emit method instead.
   */
  void EmitFatal(const Diagnostic& diagnostic);

  /*! \brief Render the errors and raise a DiagnosticError exception. */
  void Render();

  DiagnosticContextNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<DiagnosticContextNode*>(get_mutable());
  }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DiagnosticContext, ObjectRef, DiagnosticContextNode);
};

DiagnosticRenderer TerminalRenderer(std::ostream& ostream);

}  // namespace tvm
#endif  // TVM_IR_DIAGNOSTIC_H_
