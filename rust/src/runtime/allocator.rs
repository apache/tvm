#[cfg(target_env = "sgx")]
use alloc::alloc::{self, Layout};
#[cfg(not(target_env = "sgx"))]
use std::alloc::{self, Layout};

use errors::*;

const DEFAULT_ALIGN_BYTES: usize = 4;

#[derive(PartialEq, Eq)]
pub struct Allocation {
  layout: Layout,
  ptr: *mut u8,
}

impl Allocation {
  /// Allocates a chunk of memory of `size` bytes with optional alignment.
  pub fn new(size: usize, align: Option<usize>) -> Result<Self> {
    let alignment = align.unwrap_or(DEFAULT_ALIGN_BYTES);
    let layout = Layout::from_size_align(size, alignment)?;
    let ptr = unsafe { alloc::alloc(layout.clone()) };
    if ptr.is_null() {
      alloc::handle_alloc_error(layout);
    }
    Ok(Self {
      ptr: ptr,
      layout: layout,
    })
  }

  pub fn as_mut_ptr(&self) -> *mut u8 {
    self.ptr
  }

  /// Returns the size of the Allocation in bytes.
  pub fn size(&self) -> usize {
    self.layout.size()
  }

  /// Returns the byte alignment of the Allocation.
  pub fn align(&self) -> usize {
    self.layout.align()
  }
}

impl Drop for Allocation {
  fn drop(&mut self) {
    unsafe {
      alloc::dealloc(self.ptr, self.layout.clone());
    }
  }
}
