use std::{
  cell::RefCell,
  os::raw::{c_int, c_void},
  ptr,
};

use super::allocator::Allocation;
use errors::*;

struct WorkspacePool {
  workspaces: Vec<Allocation>,
  free: Vec<usize>,
  in_use: Vec<usize>,
}

impl WorkspacePool {
  fn new() -> Self {
    WorkspacePool {
      workspaces: Vec::new(),
      free: Vec::new(),
      in_use: Vec::new(),
    }
  }

  fn alloc(&mut self, size: usize) -> Result<*mut u8> {
    if self.free.len() == 0 {
      self.workspaces.push(Allocation::new(size, None)?);
      self.free.push(self.workspaces.len() - 1);
      Ok(self.workspaces[self.workspaces.len() - 1].as_mut_ptr())
    } else {
      let i = self.free.iter().fold(0, |cur_ws_idx, &idx| {
        let cur_size = self.workspaces[cur_ws_idx].size();
        let ws_size = self.workspaces[idx].size();
        if ws_size < size || ws_size > cur_size {
          cur_ws_idx
        } else {
          idx
        }
      });
      let idx = self.free.remove(i);
      self.in_use.push(idx.clone());
      Ok(self.workspaces[idx].as_mut_ptr())
    }
  }

  fn free(&mut self, ptr: *mut u8) -> Result<()> {
    let mut ws_idx = None;
    for i in 0..self.in_use.len() {
      let idx = self.in_use[i];
      if self.workspaces[idx].as_mut_ptr() == ptr {
        self.in_use.remove(i);
        ws_idx = Some(idx);
        break;
      }
    }
    Ok(
      self
        .free
        .push(ws_idx.ok_or("Tried to free nonexistent workspace.")?),
    )
  }
}

thread_local!(static WORKSPACE_POOL: RefCell<WorkspacePool> = RefCell::new(WorkspacePool::new()));

const WORKSPACE_PAGE_SIZE: usize = 4 << 10;

#[no_mangle]
pub extern "C" fn TVMBackendAllocWorkspace(
  _device_type: c_int,
  _device_id: c_int,
  size: u64,
  _dtype_code_hint: c_int,
  _dtype_bits_hint: c_int,
) -> *mut c_void {
  let nbytes = if size == 0 {
    WORKSPACE_PAGE_SIZE
  } else {
    size as usize
  };
  WORKSPACE_POOL.with(|pool_cell| {
    (match pool_cell.borrow_mut().alloc(nbytes as usize) {
      Ok(ptr) => ptr,
      Err(_) => ptr::null_mut(),
    }) as *mut c_void
  });
  return ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn TVMBackendFreeWorkspace(
  _device_type: c_int,
  _device_id: c_int,
  ptr: *mut c_void,
) -> c_int {
  WORKSPACE_POOL.with(|pool_cell| {
    (match pool_cell.borrow_mut().free(ptr as *mut u8) {
      Ok(()) => 0,
      Err(_) => -1,
    }) as c_int
  });
  return 0;
}
