use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use codespan_reporting::diagnostic::{Diagnostic as CDiagnostic, Label, Severity};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

use crate::ir::source_map::*;
use super::*;

enum StartOrEnd {
    Start,
    End,
}

enum FileSpanToByteRange {
    AsciiSource,
    Utf8 {
        /// Map character regions which are larger then 1-byte to length.
        lengths: HashMap<isize, isize>,
        source: String,
    }
}

impl FileSpanToByteRange {
    fn new(source: String) -> FileSpanToByteRange {
        let mut last_index = 0;
        let mut is_ascii = true;
        if source.is_ascii() {
            FileSpanToByteRange::AsciiSource
        } else {
            panic!()
        }

        // for (index, _) in source.char_indices() {
        //     if last_index - 1 != last_index {
        //         is_ascii = false;
        //     } else {
        //         panic!();
        //     }
        //     last_index = index;
        // }
    }
}

struct SpanToByteRange {
    map: HashMap<String, FileSpanToByteRange>
}

impl SpanToByteRange {
    fn new() -> SpanToByteRange {
        SpanToByteRange { map: HashMap::new() }
    }

    pub fn add_source(&mut self, source: Source) {
        let source_name: String = source.source_name.name.as_str().expect("foo").into();

        if self.map.contains_key(&source_name) {
            panic!()
        } else {
            let source = source.source.as_str().expect("fpp").into();
            self.map.insert(source_name, FileSpanToByteRange::new(source));
        }
    }
}

struct ByteRange<FileId> {
    file_id: FileId,
    start_pos: usize,
    end_pos: usize,
}


pub fn to_diagnostic(diag: super::Diagnostic) -> CDiagnostic<String> {
    let severity = match diag.level {
        DiagnosticLevel::Error => Severity::Error,
        DiagnosticLevel::Warning => Severity::Warning,
        DiagnosticLevel::Note => Severity::Note,
        DiagnosticLevel::Help => Severity::Help,
        DiagnosticLevel::Bug => Severity::Bug,
    };

    let file_id = "foo".into(); // diag.span.source_name;

    let message: String = diag.message.as_str().unwrap().into();
    let inner_message: String = "expected `String`, found `Nat`".into();
    let diagnostic = CDiagnostic::new(severity)
        .with_message(message)
        .with_code("EXXX")
        .with_labels(vec![
            Label::primary(file_id, 328..331).with_message(inner_message)
        ]);

    diagnostic
}

struct DiagnosticState {
    files: SimpleFiles<String, String>,
    span_map: SpanToByteRange,
}

impl DiagnosticState {
    fn new() -> DiagnosticState {
        DiagnosticState {
            files: SimpleFiles::new(),
            span_map: SpanToByteRange::new(),
        }
    }
}

fn renderer(state: &mut DiagnosticState, diag_ctx: DiagnosticContext) {
    let source_map = diag_ctx.module.source_map.clone();
        for diagnostic in diag_ctx.diagnostics.clone() {
            match source_map.source_map.get(&diagnostic.span.source_name) {
                Err(err) => panic!(),
                Ok(source) => state.span_map.add_source(source),
            }
            println!("Diagnostic: {}", diagnostic.message);
        }
}

pub fn init() -> Result<()> {
    let diag_state = Arc::new(Mutex::new(DiagnosticState::new()));
    let render_fn = move |diag_ctx: DiagnosticContext| {
        // let mut guard = diag_state.lock().unwrap();
        // renderer(&mut *guard, diag_ctx);
    };

    override_renderer(Some(render_fn))?;
    Ok(())
}
