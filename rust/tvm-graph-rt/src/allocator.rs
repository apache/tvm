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

use std::alloc::{self, Layout, LayoutError};

const DEFAULT_ALIGN_BYTES: usize = 4;

#[derive(PartialEq, Eq)]
pub struct Allocation {
    layout: Layout,
    ptr: *mut u8,
}

impl Allocation {
    /// Allocates a chunk of memory of `size` bytes with optional alignment.
    pub fn new(size: usize, align: Option<usize>) -> Result<Self, LayoutError> {
        let alignment = align.unwrap_or(DEFAULT_ALIGN_BYTES);
        let layout = Layout::from_size_align(size, alignment)?;
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Ok(Self { ptr, layout })
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

    /// Returns a view of the Allocation.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_mut_ptr(), self.size()) }
    }

    /// Returns a mutable view of the Allocation.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.size()) }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr, self.layout);
        }
    }
}
