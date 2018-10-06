use std::{
  cell::RefCell,
  os::raw::{c_int, c_void},
  ptr,
};

use super::allocator::Allocation;
use errors::*;

const WS_ALIGN: usize = 64; // taken from `kTempAllocaAlignment` in `device_api.h`

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

  fn alloc_new(&mut self, size: usize) -> Result<*mut u8> {
    self.workspaces.push(Allocation::new(size, Some(WS_ALIGN))?);
    self.in_use.push(self.workspaces.len() - 1);
    Ok(self.workspaces[self.workspaces.len() - 1].as_mut_ptr())
  }

  fn alloc(&mut self, size: usize) -> Result<*mut u8> {
    if self.free.len() == 0 {
      return self.alloc_new(size);
    }
    let idx = self
      .free
      .iter()
      .fold(None, |cur_ws_idx: Option<usize>, &idx| {
        let ws_size = self.workspaces[idx].size();
        if !ws_size >= size {
          return cur_ws_idx;
        }
        cur_ws_idx.or(Some(idx)).and_then(|cur_idx| {
          let cur_size = self.workspaces[cur_idx].size();
          Some(match ws_size <= cur_size {
            true => idx,
            false => cur_idx,
          })
        })
      });
    match idx {
      Some(idx) => {
        self.free.remove_item(&idx).unwrap();
        self.in_use.push(idx);
        Ok(self.workspaces[idx].as_mut_ptr())
      }
      None => self.alloc_new(size),
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
    pool_cell
      .borrow_mut()
      .alloc(nbytes as usize)
      .unwrap_or(ptr::null_mut()) as *mut c_void
  })
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
