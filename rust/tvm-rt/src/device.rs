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

use std::os::raw::c_void;
use std::ptr;

use crate::errors::Error;

use tvm_sys::ffi;

pub use tvm_sys::device::*;

trait DeviceExt {
    /// Checks whether the device exists or not.
    fn exist(&self) -> bool;
    fn sync(&self) -> Result<(), Error>;
    fn max_threads_per_block(&self) -> isize;
    fn warp_size(&self) -> isize;
    fn max_shared_memory_per_block(&self) -> isize;
    fn compute_version(&self) -> isize;
    fn device_name(&self) -> isize;
    fn max_clock_rate(&self) -> isize;
    fn multi_processor_count(&self) -> isize;
    fn max_thread_dimensions(&self) -> isize;
}

macro_rules! impl_device_attrs {
    ($(($attr_name:ident, $attr_kind:expr));+) => {
        $(
                fn $attr_name(&self) -> isize {
                    get_device_attr(self.device_type as i32, self.device_id as i32, 0)
                        .expect("should not fail") as isize
                }

        )+
    };
}

crate::external! {
    #[name("runtime.GetDeviceAttr")]
    fn get_device_attr(device_type: i32, device_id: i32, device_kind: i32) -> i32;
}

impl DeviceExt for Device {
    fn exist(&self) -> bool {
        let exists = get_device_attr(self.device_type as i32, self.device_id as i32, 0)
            .expect("should not fail");

        exists != 0
    }

    /// Synchronize the device stream.
    fn sync(&self) -> Result<(), Error> {
        check_call!(ffi::TVMSynchronize(
            self.device_type as i32,
            self.device_id as i32,
            ptr::null_mut() as *mut c_void
        ));
        Ok(())
    }

    impl_device_attrs!((max_threads_per_block, 1);
        (warp_size, 2);
        (max_shared_memory_per_block, 3);
        (compute_version, 4);
        (device_name, 5);
        (max_clock_rate, 6);
        (multi_processor_count, 7);
        (max_thread_dimensions, 8));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync() {
        let dev = Device::cpu(0);
        assert!(dev.sync().is_ok())
    }
}
