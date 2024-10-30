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
 * \file src/ir/diagnostic.cc
 * \brief Implementation of DiagnosticContext and friends.
 */
#include <tvm/ir/diagnostic.h>
#include <tvm/ir/source_map.h>

#include <rang.hpp>

namespace tvm {

// failed to check to argument arg0.dims[0] != 0

/* Diagnostic */
TVM_REGISTER_NODE_TYPE(DiagnosticNode);

TVM_REGISTER_GLOBAL("diagnostics.Diagnostic")
    .set_body_typed([](int level, Span span, String message) {
      return Diagnostic(static_cast<DiagnosticLevel>(level), span, message);
    });

Diagnostic::Diagnostic(DiagnosticLevel level, Span span, const std::string& message) {
  auto n = make_object<DiagnosticNode>();
  n->level = level;
  n->span = span;
  n->message = message;
  data_ = std::move(n);
}

DiagnosticBuilder Diagnostic::Bug(Span span) {
  return DiagnosticBuilder(DiagnosticLevel::kBug, span);
}

DiagnosticBuilder Diagnostic::Error(Span span) {
  return DiagnosticBuilder(DiagnosticLevel::kError, span);
}

DiagnosticBuilder Diagnostic::Warning(Span span) {
  return DiagnosticBuilder(DiagnosticLevel::kWarning, span);
}

DiagnosticBuilder Diagnostic::Note(Span span) {
  return DiagnosticBuilder(DiagnosticLevel::kNote, span);
}

DiagnosticBuilder Diagnostic::Help(Span span) {
  return DiagnosticBuilder(DiagnosticLevel::kHelp, span);
}

DiagnosticBuilder Diagnostic::Bug(ObjectRef loc) {
  return DiagnosticBuilder(DiagnosticLevel::kBug, loc);
}

DiagnosticBuilder Diagnostic::Error(ObjectRef loc) {
  return DiagnosticBuilder(DiagnosticLevel::kError, loc);
}

DiagnosticBuilder Diagnostic::Warning(ObjectRef loc) {
  return DiagnosticBuilder(DiagnosticLevel::kWarning, loc);
}

DiagnosticBuilder Diagnostic::Note(ObjectRef loc) {
  return DiagnosticBuilder(DiagnosticLevel::kNote, loc);
}

DiagnosticBuilder Diagnostic::Help(ObjectRef loc) {
  return DiagnosticBuilder(DiagnosticLevel::kHelp, loc);
}

DiagnosticBuilder Diagnostic::Bug(const Object* loc) { return Bug(GetRef<ObjectRef>(loc)); }

DiagnosticBuilder Diagnostic::Error(const Object* loc) { return Error(GetRef<ObjectRef>(loc)); }

DiagnosticBuilder Diagnostic::Note(const Object* loc) { return Note(GetRef<ObjectRef>(loc)); }

DiagnosticBuilder Diagnostic::Help(const Object* loc) { return Help(GetRef<ObjectRef>(loc)); }

/* Diagnostic Renderer */
TVM_REGISTER_NODE_TYPE(DiagnosticRendererNode);

void DiagnosticRenderer::Render(const DiagnosticContext& ctx) { (*this)->renderer(ctx); }

TVM_DLL DiagnosticRenderer::DiagnosticRenderer(
    TypedPackedFunc<void(DiagnosticContext ctx)> renderer) {
  auto n = make_object<DiagnosticRendererNode>();
  n->renderer = renderer;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("diagnostics.DiagnosticRenderer")
    .set_body_typed([](TypedPackedFunc<void(DiagnosticContext ctx)> renderer) {
      return DiagnosticRenderer(renderer);
    });

/* Diagnostic Context */
TVM_REGISTER_NODE_TYPE(DiagnosticContextNode);

void DiagnosticContext::Render() {
  (*this)->renderer.Render(*this);

  int errs = 0;
  if ((*this)->diagnostics.size()) {
    for (auto diagnostic : (*this)->diagnostics) {
      if (diagnostic->level == DiagnosticLevel::kError) {
        errs += 1;
      }
    }
  }

  if (errs) {
    (*this)->renderer = DiagnosticRenderer([](DiagnosticContext) {});
    // (*this)->diagnostics.clear();
    LOG(FATAL) << "DiagnosticError: one or more error diagnostics were "
               << "emitted, please check diagnostic render for output.";
  }
}

TVM_REGISTER_GLOBAL("diagnostics.DiagnosticRendererRender")
    .set_body_typed([](DiagnosticRenderer renderer, DiagnosticContext ctx) {
      renderer.Render(ctx);
    });

DiagnosticContext::DiagnosticContext(const IRModule& module, const DiagnosticRenderer& renderer) {
  CHECK(renderer.defined()) << "can not initialize a diagnostic renderer with a null function";
  auto n = make_object<DiagnosticContextNode>();
  n->module = module;
  n->renderer = renderer;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("diagnostics.DiagnosticContext")
    .set_body_typed([](const IRModule& module, const DiagnosticRenderer& renderer) {
      return DiagnosticContext(module, renderer);
    });

/*! \brief Emit a diagnostic. */
void DiagnosticContext::Emit(const Diagnostic& diagnostic) {
  (*this)->diagnostics.push_back(diagnostic);
}

TVM_REGISTER_GLOBAL("diagnostics.Emit")
    .set_body_typed([](DiagnosticContext ctx, const Diagnostic& diagnostic) {
      return ctx.Emit(diagnostic);
    });

TVM_REGISTER_GLOBAL("diagnostics.DiagnosticContextRender")
    .set_body_typed([](DiagnosticContext context) { return context.Render(); });

/*! \brief Emit a diagnostic. */
void DiagnosticContext::EmitFatal(const Diagnostic& diagnostic) {
  Emit(diagnostic);
  Render();
}

/* Default Terminal Renderer. */
static const char* DEFAULT_RENDERER = "diagnostics.DefaultRenderer";
static const char* OVERRIDE_RENDERER = "diagnostics.OverrideRenderer";

DiagnosticRenderer GetRenderer() {
  auto override_pf = tvm::runtime::Registry::Get(OVERRIDE_RENDERER);
  tvm::runtime::TypedPackedFunc<ObjectRef()> pf;
  if (override_pf) {
    pf = tvm::runtime::TypedPackedFunc<ObjectRef()>(*override_pf);
  } else {
    auto default_pf = tvm::runtime::Registry::Get(DEFAULT_RENDERER);
    ICHECK(default_pf != nullptr)
        << "Can not find registered function for " << DEFAULT_RENDERER << "." << std::endl
        << "Either this is an internal error or the default function was overloaded incorrectly.";
    pf = tvm::runtime::TypedPackedFunc<ObjectRef()>(*default_pf);
  }
  return Downcast<DiagnosticRenderer>(pf());
}

DiagnosticContext DiagnosticContext::Default(const IRModule& module) {
  auto renderer = GetRenderer();
  return DiagnosticContext(module, renderer);
}

TVM_REGISTER_GLOBAL("diagnostics.Default").set_body_typed([](const IRModule& module) {
  return DiagnosticContext::Default(module);
});

std::ostream& EmitDiagnosticHeader(std::ostream& out, const Span& span, DiagnosticLevel level,
                                   std::string msg) {
  rang::fg diagnostic_color = rang::fg::reset;
  std::string diagnostic_type;

  switch (level) {
    case DiagnosticLevel::kWarning: {
      diagnostic_color = rang::fg::yellow;
      diagnostic_type = "warning";
      break;
    }
    case DiagnosticLevel::kError: {
      diagnostic_color = rang::fg::red;
      diagnostic_type = "error";
      break;
    }
    case DiagnosticLevel::kBug: {
      diagnostic_color = rang::fg::blue;
      diagnostic_type = "bug";
      break;
    }
    case DiagnosticLevel::kNote: {
      diagnostic_color = rang::fg::reset;
      diagnostic_type = "note";
      break;
    }
    case DiagnosticLevel::kHelp: {
      diagnostic_color = rang::fg::reset;
      diagnostic_type = "help";
      break;
    }
  }

  out << rang::style::bold << diagnostic_color << diagnostic_type << ": " << rang::fg::reset << msg
      << std::endl
      << rang::fg::blue << " --> " << rang::fg::reset << rang::style::reset
      << span->source_name->name << ":" << span->line << ":" << span->column << std::endl;

  return out;
}

/*! \brief Generate an error message at a specific line and column with the
 * annotated message.
 *
 * The error is written directly to the `out` std::ostream.
 *
 * \param out The output ostream.
 * \param line The line at which to report a diagnostic.
 * \param line The column at which to report a diagnostic.
 * \param msg The message to attach.
 */
void ReportAt(const DiagnosticContext& context, std::ostream& out, const Span& span,
              const Diagnostic& diagnostic) {
  if (!span.defined()) {
    out << diagnostic->message << std::endl;
    return;
  }

  ICHECK(context->module->source_map.defined());
  auto it = context->module->source_map->source_map.find(span->source_name);

  // If the source name is not in the current source map, sources were not annotated.
  if (it == context->module->source_map->source_map.end()) {
    LOG(FATAL) << "The source maps are not populated for this module. "
               << "Please use `tvm.relay.transform.AnnotateSpans` to attach source maps for error "
                  "reporting.\n"
               << "Error: " << diagnostic->message;
  }

  auto source = (*it).second;
  VLOG(1) << "Source: " << std::endl << source->source;

  VLOG(1) << "ReportAt "
          << "span = " << span << " msg = " << diagnostic->message;

  auto line_text = source.GetLine(span->line);

  std::stringstream line_header_s;
  line_header_s << " " << span->line << " ";
  auto line_header = line_header_s.str();

  std::stringstream no_line_header_s;
  for (size_t i = 0; i < line_header.size(); i++) {
    no_line_header_s << " ";
  }
  auto no_line_header = no_line_header_s.str();

  EmitDiagnosticHeader(out, span, diagnostic->level, diagnostic->message)
      << no_line_header << "|  " << std::endl
      << line_header << "|  " << line_text << std::endl
      << no_line_header << "|  ";

  std::stringstream marker;
  for (size_t i = 1; i <= line_text.size(); i++) {
    if (static_cast<int>(i) >= span->column && static_cast<int>(i) < span->end_column) {
      marker << "^";
    } else {
      marker << " ";
    }
  }
  out << marker.str();
  out << std::endl;
}

// TODO(@jroesch): eventually modularize the rendering interface to provide control of how to
// format errors.
DiagnosticRenderer TerminalRenderer(std::ostream& out) {
  return DiagnosticRenderer([&](const DiagnosticContext& ctx) {
    for (auto diagnostic : ctx->diagnostics) {
      ReportAt(ctx, out, diagnostic->span, diagnostic);
    }
  });
}

TVM_REGISTER_GLOBAL(DEFAULT_RENDERER).set_body_typed([]() { return TerminalRenderer(std::cerr); });

TVM_REGISTER_GLOBAL("diagnostics.GetRenderer").set_body_typed([]() { return GetRenderer(); });

TVM_REGISTER_GLOBAL("diagnostics.ClearRenderer").set_body_typed([]() {
  tvm::runtime::Registry::Remove(OVERRIDE_RENDERER);
});

}  // namespace tvm
