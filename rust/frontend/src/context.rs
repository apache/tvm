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

//! Provides [`TVMContext`] and related device specific queries.
//!
//! Create a new context by device type (cpu is 1) and device id.
//!
//! # Example
//!
//! ```
//! # use tvm_frontend::{TVMDeviceType, TVMContext};
//! let cpu = TVMDeviceType::from("cpu");
//! let ctx = TVMContext::new(cpu , 0);
//! let cpu0 = TVMContext::cpu(0);
//! assert_eq!(ctx, cpu0);
//! ```
//!
//! Or from a supported device name.
//!
//! ```
//! use tvm_frontend::TVMContext;
//! let cpu0 = TVMContext::from("cpu");
//! println!("{}", cpu0);
//! ```

use std::{
    convert::TryInto,
    fmt::{self, Display, Formatter},
    os::raw::c_void,
    ptr,
};

use failure::Error;

use tvm_common::ffi;

use crate::{function, TVMArgValue};

/// Device type can be from a supported device name. See the supported devices
/// in [TVM](https://github.com/apache/incubator-tvm).
///
/// ## Example
///
/// ```
/// use tvm_frontend::TVMDeviceType;
/// let cpu = TVMDeviceType::from("cpu");
/// println!("device is: {}", cpu);
///```

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TVMDeviceType(pub i64);

impl Default for TVMDeviceType {
    /// default device is cpu.
    fn default() -> Self {
        TVMDeviceType(1)
    }
}

impl From<TVMDeviceType> for ffi::DLDeviceType {
    fn from(device_type: TVMDeviceType) -> Self {
        match device_type.0 {
            1 => ffi::DLDeviceType_kDLCPU,
            2 => ffi::DLDeviceType_kDLGPU,
            3 => ffi::DLDeviceType_kDLCPUPinned,
            4 => ffi::DLDeviceType_kDLOpenCL,
            7 => ffi::DLDeviceType_kDLVulkan,
            8 => ffi::DLDeviceType_kDLMetal,
            9 => ffi::DLDeviceType_kDLVPI,
            10 => ffi::DLDeviceType_kDLROCM,
            12 => ffi::DLDeviceType_kDLExtDev,
            _ => panic!("device type not found!"),
        }
    }
}

impl From<ffi::DLDeviceType> for TVMDeviceType {
    fn from(device_type: ffi::DLDeviceType) -> Self {
        match device_type {
            ffi::DLDeviceType_kDLCPU => TVMDeviceType(1),
            ffi::DLDeviceType_kDLGPU => TVMDeviceType(2),
            ffi::DLDeviceType_kDLCPUPinned => TVMDeviceType(3),
            ffi::DLDeviceType_kDLOpenCL => TVMDeviceType(4),
            ffi::DLDeviceType_kDLVulkan => TVMDeviceType(7),
            ffi::DLDeviceType_kDLMetal => TVMDeviceType(8),
            ffi::DLDeviceType_kDLVPI => TVMDeviceType(9),
            ffi::DLDeviceType_kDLROCM => TVMDeviceType(10),
            ffi::DLDeviceType_kDLExtDev => TVMDeviceType(12),
            _ => panic!("device type not found!"),
        }
    }
}

impl Display for TVMDeviceType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TVMDeviceType(1) => "cpu",
                TVMDeviceType(2) => "gpu",
                TVMDeviceType(3) => "cpu_pinned",
                TVMDeviceType(4) => "opencl",
                TVMDeviceType(8) => "meta",
                TVMDeviceType(9) => "vpi",
                TVMDeviceType(10) => "rocm",
                TVMDeviceType(_) => "rpc",
            }
        )
    }
}

impl<'a> From<&'a str> for TVMDeviceType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "cpu" => TVMDeviceType(1),
            "llvm" => TVMDeviceType(1),
            "stackvm" => TVMDeviceType(1),
            "gpu" => TVMDeviceType(2),
            "cuda" => TVMDeviceType(2),
            "nvptx" => TVMDeviceType(2),
            "cl" => TVMDeviceType(4),
            "opencl" => TVMDeviceType(4),
            "metal" => TVMDeviceType(8),
            "vpi" => TVMDeviceType(9),
            "rocm" => TVMDeviceType(10),
            _ => panic!("{:?} not supported!", type_str),
        }
    }
}

impl<'a> From<&TVMDeviceType> for TVMArgValue<'a> {
    fn from(dev: &TVMDeviceType) -> Self {
        Self::Int(dev.0)
    }
}

/// Represents the underlying device context. Default is cpu.
///
/// ## Examples
///
/// ```
/// use tvm_frontend::TVMContext;
/// let ctx = TVMContext::from("cpu");
/// assert!(ctx.exist());
///
/// ```
///
/// It is possible to query the underlying context as follows
///
/// ```
/// # use tvm_frontend::TVMContext;
/// # let ctx = TVMContext::from("cpu");
/// println!("maximun threads per block: {}", ctx.exist());
/// ```
//  TODO: add example back for GPU
// println!("compute version: {}", ctx.compute_version());
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TVMContext {
    /// Supported device types
    pub device_type: TVMDeviceType,
    /// Device id
    pub device_id: i32,
}

impl TVMContext {
    /// Creates context from device type and id.
    pub fn new(device_type: TVMDeviceType, device_id: i32) -> Self {
        TVMContext {
            device_type,
            device_id,
        }
    }
}

macro_rules! impl_ctxs {
    ($(($ctx:ident, $dldevt:expr));+) => {
        $(
            impl TVMContext {
                pub fn $ctx(device_id: i32) -> Self {
                    Self::new(TVMDeviceType($dldevt), device_id)
                }
            }
        )+
    };
}

impl_ctxs!((cpu, 1);
            (gpu, 2);
            (nvptx, 2);
            (cuda, 2);
            (cpu_pinned, 3);
            (cl, 4);
            (opencl, 4);
            (metal, 8);
            (vpi, 9);
            (rocm, 10);
            (opengl, 11);
            (ext_dev, 12));

impl<'a> From<&'a str> for TVMContext {
    fn from(target: &str) -> Self {
        TVMContext::new(TVMDeviceType::from(target), 0)
    }
}

impl TVMContext {
    /// Checks whether the context exists or not.
    pub fn exist(&self) -> bool {
        let func = function::Function::get("runtime.GetDeviceAttr")
            .expect("TVM FFI functions must always be registered.");
        let dt = self.device_type.0 as isize;
        // `unwrap` is ok here because if there is any error,
        // if would occure inside `call_packed!`
        let ret: i64 = call_packed!(func, dt, self.device_id, 0)
            .unwrap()
            .try_into()
            .unwrap();
        ret != 0
    }

    /// Synchronize the context stream.
    pub fn sync(&self) -> Result<(), Error> {
        check_call!(ffi::TVMSynchronize(
            self.device_type.0 as i32,
            self.device_id as i32,
            ptr::null_mut() as *mut c_void
        ));
        Ok(())
    }
}

macro_rules! impl_device_attrs {
    ($(($attr_name:ident, $attr_kind:expr));+) => {
        $(
            impl TVMContext {
                pub fn $attr_name(&self) -> isize {
                    let func = function::Function::get("runtime.GetDeviceAttr")
                        .expect("TVM FFI functions must always be registered.");
                    let dt = self.device_type.0 as isize;
                    // TODO(@jroesch): these functions CAN and WILL return NULL
                    // we should make these optional or somesuch to handle this.
                    // `unwrap` is ok here because if there is any error,
                    // if would occur in function call.
                    function::Builder::from(func)
                        .arg(dt)
                        .arg(self.device_id as isize)
                        .arg($attr_kind)
                        .invoke()
                        .unwrap()
                        .try_into()
                        .unwrap()
                }
            }
        )+
    };
}

impl_device_attrs!((max_threads_per_block, 1);
                (warp_size, 2);
                (max_shared_memory_per_block, 3);
                (compute_version, 4);
                (device_name, 5);
                (max_clock_rate, 6);
                (multi_processor_count, 7);
                (max_thread_dimensions, 8));

impl From<ffi::DLContext> for TVMContext {
    fn from(ctx: ffi::DLContext) -> Self {
        TVMContext {
            device_type: TVMDeviceType::from(ctx.device_type),
            device_id: ctx.device_id,
        }
    }
}

impl From<TVMContext> for ffi::DLContext {
    fn from(ctx: TVMContext) -> Self {
        ffi::DLContext {
            device_type: ctx.device_type.into(),
            device_id: ctx.device_id as i32,
        }
    }
}

impl Display for TVMContext {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.device_type, self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context() {
        let ctx = TVMContext::cpu(0);
        println!("ctx: {}", ctx);
        let default_ctx = TVMContext::new(TVMDeviceType(1), 0);
        assert_eq!(ctx.clone(), default_ctx);
        assert_ne!(ctx, TVMContext::gpu(0));

        let str_ctx = TVMContext::new(TVMDeviceType::from("gpu"), 0);
        assert_eq!(str_ctx.clone(), str_ctx);
        assert_ne!(str_ctx, TVMContext::new(TVMDeviceType::from("cpu"), 0));
    }

    #[test]
    fn sync() {
        let ctx = TVMContext::cpu(0);
        assert!(ctx.sync().is_ok())
    }
}
