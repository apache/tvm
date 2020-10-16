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

/// A TVM diagnostics renderer which uses the Rust `codespan`
/// library to produce error messages.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use codespan_reporting::diagnostic::{Diagnostic as CDiagnostic, Label, Severity};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self, ColorArg};

use super::*;
use crate::ir::source_map::*;

struct ByteRange<FileId> {
    file_id: FileId,
    start_pos: usize,
    end_pos: usize,
}

enum FileSpanToByteRange {
    AsciiSource(Vec<usize>),
    Utf8 {
        /// Map character regions which are larger then 1-byte to length.
        lengths: HashMap<isize, isize>,
        source: String,
    },
}

impl FileSpanToByteRange {
    fn new(source: String) -> FileSpanToByteRange {
        let mut last_index = 0;
        let mut is_ascii = true;
        if source.is_ascii() {
            let line_lengths = source.lines().map(|line| line.len()).collect();
            FileSpanToByteRange::AsciiSource(line_lengths)
        } else {
            panic!()
        }
    }

    fn lookup(&self, span: &Span) -> ByteRange<String> {
        use FileSpanToByteRange::*;

        let source_name: String = span.source_name.name.as_str().unwrap().into();

        match self {
            AsciiSource(ref line_lengths) => {
                let start_pos = (&line_lengths[0..(span.line - 1) as usize])
                    .into_iter()
                    .sum::<usize>()
                    + (span.column) as usize;
                let end_pos = (&line_lengths[0..(span.end_line - 1) as usize])
                    .into_iter()
                    .sum::<usize>()
                    + (span.end_column) as usize;
                ByteRange {
                    file_id: source_name,
                    start_pos,
                    end_pos,
                }
            }
            _ => panic!(),
        }
    }
}

struct SpanToByteRange {
    map: HashMap<String, FileSpanToByteRange>,
}

impl SpanToByteRange {
    fn new() -> SpanToByteRange {
        SpanToByteRange {
            map: HashMap::new(),
        }
    }

    pub fn add_source(&mut self, source: Source) {
        let source_name: String = source.source_name.name.as_str().expect("foo").into();

        if self.map.contains_key(&source_name) {
            panic!()
        } else {
            let source = source.source.as_str().expect("fpp").into();
            self.map
                .insert(source_name, FileSpanToByteRange::new(source));
        }
    }

    pub fn lookup(&self, span: &Span) -> ByteRange<String> {
        let source_name: String = span.source_name.name.as_str().expect("foo").into();

        match self.map.get(&source_name) {
            Some(file_span_to_bytes) => file_span_to_bytes.lookup(span),
            None => panic!(),
        }
    }
}

struct DiagnosticState {
    files: SimpleFiles<String, String>,
    span_map: SpanToByteRange,
    // todo unify wih source name
    source_to_id: HashMap<String, usize>,
}

impl DiagnosticState {
    fn new() -> DiagnosticState {
        DiagnosticState {
            files: SimpleFiles::new(),
            span_map: SpanToByteRange::new(),
            source_to_id: HashMap::new(),
        }
    }

    fn add_source(&mut self, source: Source) {
        let source_str: String = source.source.as_str().unwrap().into();
        let source_name: String = source.source_name.name.as_str().unwrap().into();
        self.span_map.add_source(source);
        let file_id = self.files.add(source_name.clone(), source_str);
        self.source_to_id.insert(source_name, file_id);
    }

    fn to_diagnostic(&self, diag: super::Diagnostic) -> CDiagnostic<usize> {
        let severity = match diag.level {
            DiagnosticLevel::Error => Severity::Error,
            DiagnosticLevel::Warning => Severity::Warning,
            DiagnosticLevel::Note => Severity::Note,
            DiagnosticLevel::Help => Severity::Help,
            DiagnosticLevel::Bug => Severity::Bug,
        };

        let source_name: String = diag.span.source_name.name.as_str().unwrap().into();
        let file_id = *self.source_to_id.get(&source_name).unwrap();

        let message: String = diag.message.as_str().unwrap().into();

        let byte_range = self.span_map.lookup(&diag.span);

        let diagnostic = CDiagnostic::new(severity)
            .with_message(message)
            .with_code("EXXX")
            .with_labels(vec![Label::primary(
                file_id,
                byte_range.start_pos..byte_range.end_pos,
            )]);

        diagnostic
    }
}

fn renderer(state: &mut DiagnosticState, diag_ctx: DiagnosticContext) {
    let source_map = diag_ctx.module.source_map.clone();
    let writer = StandardStream::stderr(ColorChoice::Always);
    let config = codespan_reporting::term::Config::default();
    for diagnostic in diag_ctx.diagnostics.clone() {
        match source_map.source_map.get(&diagnostic.span.source_name) {
            Err(err) => panic!(err),
            Ok(source) => {
                state.add_source(source);
                let diagnostic = state.to_diagnostic(diagnostic);
                term::emit(&mut writer.lock(), &config, &state.files, &diagnostic).unwrap();
            }
        }
    }
}

pub fn init() -> Result<()> {
    let diag_state = Arc::new(Mutex::new(DiagnosticState::new()));
    let render_fn = move |diag_ctx: DiagnosticContext| {
        let mut guard = diag_state.lock().unwrap();
        renderer(&mut *guard, diag_ctx);
    };

    override_renderer(Some(render_fn))?;
    Ok(())
}
