#[derive(Debug, Fail)]
pub enum GraphFormatError {
    #[fail(display = "Could not parse graph json")]
    Parse(#[fail(cause)] failure::Error),
    #[fail(display = "Could not parse graph params")]
    Params,
    #[fail(display = "{} is missing attr: {}", 0, 1)]
    MissingAttr(String, String),
    #[fail(display = "Missing field: {}", 0)]
    MissingField(&'static str),
    #[fail(display = "Invalid DLType: {}", 0)]
    InvalidDLType(String),
}

// #[cfg(target_env = "sgx")]
// use alloc::alloc;
// #[cfg(not(target_env = "sgx"))]
// use std::alloc;
// use std::num;
//
// use ndarray;
// use serde_json;

// error_chain! {
//   errors {
//     GraphFormatError(msg: String) {
//       description("unable to load graph")
//       display("could not load graph json: {}", msg)
//     }
//
//     LoadGraphParamsError(msg: String) {
//       description("unable to load graph params")
//       display("could not load graph params: {}", msg)
//     }
//   }
//   foreign_links {
//     Alloc(alloc::AllocErr);
//     GraphDeserialize(serde_json::Error);
//     ParseInt(num::ParseIntError);
//     ShapeError(ndarray::ShapeError);
//     CommonError(tvm_common::errors::Error);
//   }
// }
//
// impl From<alloc::LayoutErr> for Error {
//     fn from(_err: alloc::LayoutErr) -> Error {
//         Error::from_kind(ErrorKind::Msg("Layout error".to_string()))
//     }
// }
