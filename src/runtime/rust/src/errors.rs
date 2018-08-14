use std::{alloc, num};

use ndarray;
use serde_json;

error_chain! {
  errors {
    TryFromTVMRetValueError(expected: String, actual: i64) {
      description("mismatched types while downcasting TVMRetValue")
      display("invalid downcast: expected `{}` but was `{}`", expected, actual)
    }

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
    Layout(alloc::LayoutErr);
    GraphDeserialize(serde_json::Error);
    ParseInt(num::ParseIntError);
    ShapeError(ndarray::ShapeError);
  }
}
