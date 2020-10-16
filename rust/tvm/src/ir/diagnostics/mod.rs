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

use super::module::IRModule;
use super::span::*;
use crate::runtime::function::Result;
use crate::runtime::object::{Object, ObjectPtr, ObjectRef};
use crate::runtime::{
    array::Array,
    function::{self, Function, ToFunction},
    string::String as TString,
};
/// The diagnostic interface to TVM, used for reporting and rendering
/// diagnostic information by the compiler. This module exposes
/// three key abstractions: a Diagnostic, the DiagnosticContext,
/// and the DiagnosticRenderer.
use tvm_macros::{external, Object};

pub mod codespan;

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

    #[name("diagnostics.ClearRenderer")]
    fn clear_renderer() -> ();
}

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
#[repr(C)]
#[derive(Object)]
#[ref_name = "DiagnosticRenderer"]
#[type_key = "DiagnosticRenderer"]
/// A diagnostic renderer, which given a diagnostic context produces a "rendered"
/// form of the diagnostics for either human or computer consumption.
pub struct DiagnosticRendererNode {
    /// The base type.
    pub base: Object,
    // TODO(@jroesch): we can't easily exposed packed functions due to
    // memory layout
    // missing field here
}

//     def render(self, ctx):
//         """
//         Render the provided context.

//         Params
//         ------
//         ctx: DiagnosticContext
//             The diagnostic context to render.
//         """
//         return _ffi_api.DiagnosticRendererRender(self, ctx

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

/// A diagnostic context which records active errors
/// and contains a renderer.
impl DiagnosticContext {
    pub fn new<F>(module: IRModule, render_func: F) -> DiagnosticContext
    where
        F: Fn(DiagnosticContext) -> () + 'static,
    {
        let renderer = diagnostic_renderer(render_func.to_function()).unwrap();
        let node = DiagnosticContextNode {
            base: Object::base_object::<DiagnosticContextNode>(),
            module,
            diagnostics: Array::from_vec(vec![]).unwrap(),
            renderer,
        };
        DiagnosticContext(Some(ObjectPtr::new(node)))
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

/// Override the global diagnostics renderer.
// render_func: Option[Callable[[DiagnosticContext], None]]
//     If the render_func is None it will remove the current custom renderer
//     and return to default behavior.
fn override_renderer<F>(opt_func: Option<F>) -> Result<()>
where
    F: Fn(DiagnosticContext) -> () + 'static,
{
    match opt_func {
        None => clear_renderer(),
        Some(func) => {
            let func = func.to_function();
            let render_factory = move || diagnostic_renderer(func.clone()).unwrap();

            function::register_override(render_factory, "diagnostics.OverrideRenderer", true)?;

            Ok(())
        }
    }
}
