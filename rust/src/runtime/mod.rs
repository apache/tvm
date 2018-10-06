mod allocator;
mod array;
mod module;
#[macro_use]
mod packed_func;
mod graph;
#[cfg(target_env = "sgx")]
#[macro_use]
pub mod sgx;
mod threading;
mod workspace;

use std::os::raw::c_char;

pub use self::{array::*, graph::*, module::*, packed_func::*, threading::*, workspace::*};

#[no_mangle]
pub extern "C" fn TVMAPISetLastError(cmsg: *const c_char) {
  #[cfg(not(target_env = "sgx"))]
  unsafe {
    panic!(std::ffi::CStr::from_ptr(cmsg).to_str().unwrap());
  }
  #[cfg(target_env = "sgx")]
  ocall_packed!("__sgx_set_last_error__", cmsg);
}
