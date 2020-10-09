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

/// The diagnostic interface to TVM, used for reporting and rendering
/// diagnostic information by the compiler. This module exposes
/// three key abstractions: a Diagnostic, the DiagnosticContext,
/// and the DiagnosticRenderer.

use tvm_macros::{Object, external};
use super::module::IRModule;
use crate::runtime::{function::{Function, Typed}, array::Array, string::String as TString};
use crate::runtime::object::{Object, ObjectRef};
use crate::runtime::function::Result;
use super::span::Span;

type SourceName = ObjectRef;

/// The diagnostic level, controls the printing of the message.
#[repr(C)]
pub enum DiagnosticLevel {
    Bug = 10,
    Error = 20,
    Warning = 30,
    Note = 40,
    Help = 50,
}

/// A compiler diagnostic.
#[repr(C)]
#[derive(Object)]
#[ref_name = "Diagnostic"]
#[type_key = "Diagnostic"]
pub struct DiagnosticNode {
    pub base: Object,
    /// The level.
    pub level: DiagnosticLevel,
    /// The span at which to report an error.
    pub span: Span,
    /// The diagnostic message.
    pub message: TString,
}

impl Diagnostic {
    pub fn new(level: DiagnosticLevel, span: Span, message: TString) {
        todo!()
    }

    pub fn bug(span: Span) -> DiagnosticBuilder {
        todo!()
    }

    pub fn error(span: Span) -> DiagnosticBuilder {
        todo!()
    }

    pub fn warning(span: Span) -> DiagnosticBuilder {
        todo!()
    }

    pub fn note(span: Span) -> DiagnosticBuilder {
        todo!()
    }

    pub fn help(span: Span) -> DiagnosticBuilder {
        todo!()
    }
}

/// A wrapper around std::stringstream to build a diagnostic.
pub struct DiagnosticBuilder {
    /// The level.
    pub level: DiagnosticLevel,

    /// The source name.
    pub source_name: SourceName,

    /// The span of the diagnostic.
    pub span: Span,
}

//   /*! \brief Display diagnostics in a given display format.
//    *
//    * A diagnostic renderer is responsible for converting the
//    * raw diagnostics into consumable output.
//    *
//    * For example the terminal renderer will render a sequence
//    * of compiler diagnostics to std::out and std::err in
//    * a human readable form.
//    */
//   class DiagnosticRendererNode : public Object {
//    public:
//     TypedPackedFunc<void(DiagnosticContext ctx)> renderer;

//     // override attr visitor
//     void VisitAttrs(AttrVisitor* v) {}

//     static constexpr const char* _type_key = "DiagnosticRenderer";
//     TVM_DECLARE_FINAL_OBJECT_INFO(DiagnosticRendererNode, Object);
//   };

//   class DiagnosticRenderer : public ObjectRef {
//    public:
//     TVM_DLL DiagnosticRenderer(TypedPackedFunc<void(DiagnosticContext ctx)> render);
//     TVM_DLL DiagnosticRenderer()
//         : DiagnosticRenderer(TypedPackedFunc<void(DiagnosticContext ctx)>()) {}

//     void Render(const DiagnosticContext& ctx);

//     DiagnosticRendererNode* operator->() {
//       CHECK(get() != nullptr);
//       return static_cast<DiagnosticRendererNode*>(get_mutable());
//     }

//     TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DiagnosticRenderer, ObjectRef, DiagnosticRendererNode);
//   };

// @tvm._ffi.register_object("DiagnosticRenderer")
// class DiagnosticRenderer(Object):
//     """
//     A diagnostic renderer, which given a diagnostic context produces a "rendered"
//     form of the diagnostics for either human or computer consumption.
//     """

//     def __init__(self, render_func):
//         self.__init_handle_by_constructor__(_ffi_api.DiagnosticRenderer, render_func)

//     def render(self, ctx):
//         """
//         Render the provided context.

//         Params
//         ------
//         ctx: DiagnosticContext
//             The diagnostic context to render.
//         """
//         return _ffi_api.DiagnosticRendererRender(self, ctx
pub type DiagnosticRenderer = ObjectRef;

#[repr(C)]
#[derive(Object)]
#[ref_name = "DiagnosticContext"]
#[type_key = "DiagnosticContext"]
/// A diagnostic context for recording errors against a source file.
pub struct DiagnosticContextNode {
    // The base type.
    pub base: Object,

    /// The Module to report against.
    pub module: IRModule,

    /// The set of diagnostics to report.
    pub diagnostics: Array<Diagnostic>,

    /// The renderer set for the context.
    pub renderer: DiagnosticRenderer,
}

// Get the the diagnostic renderer.
external! {
    #[name("node.ArrayGetItem")]
    fn get_renderer() -> DiagnosticRenderer;

    #[name("diagnostics.DiagnosticRenderer")]
    fn diagnostic_renderer(func: Function) -> DiagnosticRenderer;

    #[name("diagnostics.Emit")]
    fn emit(ctx: DiagnosticContext, diagnostic: Diagnostic) -> ();

    #[name("diagnostics.DiagnosticContextRender")]
    fn diagnostic_context_render(ctx: DiagnosticContext) -> ();
}

/// A diagnostic context which records active errors
/// and contains a renderer.
impl DiagnosticContext {
    pub fn new(module: IRModule, renderer: DiagnosticRenderer) {
        todo!()
    }

    pub fn default(module: IRModule) -> DiagnosticContext {
        todo!()
    }

    /// Emit a diagnostic.
    pub fn emit(&mut self, diagnostic: Diagnostic) -> Result<()> {
        emit(self.clone(), diagnostic)
    }

    /// Render the errors and raise a DiagnosticError exception.
    pub fn render(&mut self) -> Result<()> {
        diagnostic_context_render(self.clone())
    }

    /// Emit a diagnostic and then immediately attempt to render all errors.
    pub fn emit_fatal(&mut self, diagnostic: Diagnostic) -> Result<()> {
        self.emit(diagnostic)?;
        self.render()?;
        Ok(())
    }
}

// Sets a custom renderer for diagnostics.

// Params
// ------
// render_func: Option[Callable[[DiagnosticContext], None]]
//     If the render_func is None it will remove the current custom renderer
//     and return to default behavior.
fn override_renderer<F>(opt_func: Option<F>) -> Result<()>
where F: Fn(DiagnosticContext) -> ()
{
    todo!()
    // fn ()
    // diagnostic_renderer(func)
    // if render_func:

    //     def _render_factory():
    //         return DiagnosticRenderer(render_func)

    //     register_func("diagnostics.OverrideRenderer", _render_factory, override=True)
    // else:
    //     _ffi_api.ClearRenderer()
}
