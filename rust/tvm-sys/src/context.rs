
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

//! Provides [`Context`] and related device queries.
//!
//! Create a new context for device type and device id.
//!
//! # Example
//!
//! ```
//! # use tvm_sys::{DeviceType, Context};
//! let cpu = DeviceType::from("cpu");
//! let ctx = Context::new(cpu , 0);
//! let cpu0 = Context::cpu(0);
//! assert_eq!(ctx, cpu0);
//! ```
//!
//! Or from a supported device name.
//!
//! ```
//! use tvm_rt::Context;
//! let cpu0 = Context::from("cpu");
//! println!("{}", cpu0);
//! ```

use crate::ffi::{self, *};
use crate::packed_func::ArgValue;

use std::str::FromStr;
use thiserror::Error;

use std::{
    fmt::{self, Display, Formatter},
};

use anyhow::Result;

/// Device type can be from a supported device name. See the supported devices
/// in [TVM](https://github.com/apache/incubator-tvm).
///
/// ## Example
///
/// ```
/// use tvm_rt::TVMDeviceType;
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

impl<'a> From<&TVMDeviceType> for ArgValue<'a> {
    fn from(dev: &TVMDeviceType) -> Self {
        Self::Int(dev.0)
    }
}

 #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Context {
    pub device_type: TVMDeviceType,
    pub device_id: usize,
}

impl Context {
    pub fn new(device_type: TVMDeviceType, device_id: usize) -> Context {
        Context { device_type, device_id }
    }
}

impl<'a> From<&'a Context> for DLContext {
    fn from(ctx: &'a Context) -> Self {
        Self {
            device_type: ctx.device_type.into(),
            device_id: ctx.device_id as i32,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            device_type: DLDeviceType_kDLCPU.into(),
            device_id: 0,
        }
    }
}

#[derive(Debug, Error)]
#[error("unsupported device: {0}")]
pub struct UnsupportedDeviceError(String);

macro_rules! impl_tvm_context {
    ( $( $dev_type:ident : [ $( $dev_name:ident ),+ ] ),+ ) => {
        /// Creates a Context from a string (e.g., "cpu", "gpu", "ext_dev")
        impl FromStr for Context {
            type Err = UnsupportedDeviceError;
            fn from_str(type_str: &str) -> Result<Self, Self::Err> {
                Ok(Self {
                    device_type: match type_str {
                         $( $(  stringify!($dev_name)  )|+ => $dev_type.into()),+,
                        _ => return Err(UnsupportedDeviceError(type_str.to_string())),
                    },
                    device_id: 0,
                })
            }
        }

        impl Context {
            $(
                $(
                    pub fn $dev_name(device_id: usize) -> Self {
                        Self {
                            device_type: $dev_type.into(),
                            device_id: device_id,
                        }
                    }
                )+
            )+
        }
    };
}

impl_tvm_context!(
    DLDeviceType_kDLCPU: [cpu, llvm, stackvm],
    DLDeviceType_kDLGPU: [gpu, cuda, nvptx],
    DLDeviceType_kDLOpenCL: [cl],
    DLDeviceType_kDLMetal: [metal],
    DLDeviceType_kDLVPI: [vpi],
    DLDeviceType_kDLROCM: [rocm],
    DLDeviceType_kDLExtDev: [ext_dev]
);

impl<'a> From<&'a str> for Context {
    fn from(target: &str) -> Self {
        Context::new(TVMDeviceType::from(target), 0)
    }
}


impl From<ffi::DLContext> for Context {
    fn from(ctx: ffi::DLContext) -> Self {
        Context {
            device_type: TVMDeviceType::from(ctx.device_type),
            device_id: ctx.device_id as usize,
        }
    }
}

impl From<Context> for ffi::DLContext {
    fn from(ctx: Context) -> Self {
        ffi::DLContext {
            device_type: ctx.device_type.into(),
            device_id: ctx.device_id as i32,
        }
    }
}

impl Display for Context {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.device_type, self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context() {
        let ctx = Context::cpu(0);
        println!("ctx: {}", ctx);
        let default_ctx = Context::new(TVMDeviceType(1), 0);
        assert_eq!(ctx.clone(), default_ctx);
        assert_ne!(ctx, Context::gpu(0));

        let str_ctx = Context::new(TVMDeviceType::from("gpu"), 0);
        assert_eq!(str_ctx.clone(), str_ctx);
        assert_ne!(str_ctx, Context::new(TVMDeviceType::from("cpu"), 0));
    }

    #[test]
    fn sync() {
        let ctx = Context::cpu(0);
        assert!(ctx.sync().is_ok())
    }
}
