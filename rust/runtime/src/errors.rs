#[cfg(target_env = "sgx")]
use alloc::alloc;
#[cfg(not(target_env = "sgx"))]
use std::alloc;
use std::num;

use crate::common::errors as common_errors;
use ndarray;
use serde_json;

error_chain! {
  errors {
    GraphFormatError(msg: String) {
      description("unable to load graph")
      display("could not load graph json: {}", msg)
    }

    LoadGraphParamsError(msg: String) {
      description("unable to load graph params")
      display("could not load graph params: {}", msg)
    }
  }
  foreign_links {
    Alloc(alloc::AllocErr);
    GraphDeserialize(serde_json::Error);
    ParseInt(num::ParseIntError);
    ShapeError(ndarray::ShapeError);
    CommonError(common_errors::Error);
  }
}

impl From<alloc::LayoutErr> for Error {
    fn from(_err: alloc::LayoutErr) -> Error {
        Error::from_kind(ErrorKind::Msg("Layout error".to_string()))
    }
}
