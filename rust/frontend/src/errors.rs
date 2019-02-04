//! This module implements TVM custom [`Error`], [`ErrorKind`] and [`Result`] types.

use std::{ffi, option};

use crate::{common_errors, rust_ndarray};

error_chain! {
    errors {
        EmptyArray {
            description("cannot convert from an empty array")
        }

        NullHandle(name: String) {
            description("null handle")
            display("requested `{}` handle is null", name)
        }

        FunctionNotSet {
            description("packed function not set")
            display("packed function was not set in `function::Builder` or does not exist in the global registry")
        }

        FunctionNotFound(name: String) {
            description("packed function not found")
            display("packed function `{}` does not exist in the global registry", name)
        }

        TypeMismatch(expected: String, found: String) {
            description("type mismatch!")
            display("expected type `{}`, but found `{}`", expected, found)
        }

        MissingShapeError {
            description("ndarray `shape()` returns `None`")
            display("called `Option::unwrap()` on a `None` value")
        }

        AtMostOneReturn {
            description("TVM functions accept at most one return value")
        }

    }

    foreign_links {
        ShapeError(rust_ndarray::ShapeError);
        NulError(ffi::NulError);
        IntoStringError(ffi::IntoStringError);
        CommonError(common_errors::Error);
        IoError(std::io::Error);
    }
}

impl From<option::NoneError> for Error {
    fn from(_err: option::NoneError) -> Self {
        ErrorKind::MissingShapeError.into()
    }
}
