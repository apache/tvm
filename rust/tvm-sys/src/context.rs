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
//! use tvm_sys::Context;
//! let cpu0 = Context::from("cpu");
//! println!("{}", cpu0);
//! ```

use std::convert::TryFrom;
use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::ffi::{self, *};
use crate::packed_func::{ArgValue, RetValue};

use anyhow::Result;
use enumn::N;
use thiserror::Error;

/// Device type represents the set of devices supported by
/// [TVM](https://github.com/apache/incubator-tvm).
///
/// ## Example
///
/// ```
/// use tvm_sys::DeviceType;
/// let cpu = DeviceType::from("cpu");
/// println!("device is: {}", cpu);
///```

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, N)]
#[repr(i64)]
pub enum DeviceType {
    CPU = 1,
    GPU,
    CPUPinned,
    OpenCL,
    Vulkan,
    Metal,
    VPI,
    ROCM,
    ExtDev,
}

impl Default for DeviceType {
    /// default device is cpu.
    fn default() -> Self {
        DeviceType::CPU
    }
}

impl From<DeviceType> for ffi::DLDeviceType {
    fn from(device_type: DeviceType) -> Self {
        device_type as Self
    }
}

impl From<ffi::DLDeviceType> for DeviceType {
    fn from(device_type: ffi::DLDeviceType) -> Self {
        Self::n(device_type as _).expect("invalid enumeration value for ffi::DLDeviceType")
    }
}

impl Display for DeviceType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DeviceType::CPU => "cpu",
                DeviceType::GPU => "gpu",
                DeviceType::CPUPinned => "cpu_pinned",
                DeviceType::OpenCL => "opencl",
                DeviceType::Vulkan => "vulkan",
                DeviceType::Metal => "metal",
                DeviceType::VPI => "vpi",
                DeviceType::ROCM => "rocm",
                DeviceType::ExtDev => "ext_device",
                // DeviceType(_) => "rpc",
            }
        )
    }
}

impl<'a> From<&'a str> for DeviceType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "cpu" => DeviceType::CPU,
            "llvm" => DeviceType::CPU,
            "stackvm" => DeviceType::CPU,
            "gpu" => DeviceType::GPU,
            "cuda" => DeviceType::GPU,
            "nvptx" => DeviceType::GPU,
            "cl" => DeviceType::OpenCL,
            "opencl" => DeviceType::OpenCL,
            "metal" => DeviceType::Metal,
            "vpi" => DeviceType::VPI,
            "rocm" => DeviceType::ROCM,
            _ => panic!("{:?} not supported!", type_str),
        }
    }
}

impl<'a> From<&DeviceType> for ArgValue<'a> {
    fn from(dev: &DeviceType) -> Self {
        Self::Int(*dev as _)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Context {
    pub device_type: DeviceType,
    pub device_id: usize,
}

impl Context {
    pub fn new(device_type: DeviceType, device_id: usize) -> Context {
        Context {
            device_type,
            device_id,
        }
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
        Context::new(DeviceType::from(target), 0)
    }
}

impl From<ffi::DLContext> for Context {
    fn from(ctx: ffi::DLContext) -> Self {
        Context {
            device_type: DeviceType::from(ctx.device_type),
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

impl From<Context> for RetValue {
    fn from(ret_value: Context) -> RetValue {
        RetValue::Context(ret_value.into())
    }
}

impl TryFrom<RetValue> for Context {
    type Error = anyhow::Error;
    fn try_from(ret_value: RetValue) -> anyhow::Result<Context> {
        match ret_value {
            RetValue::Context(dt) => Ok(dt.into()),
            // TODO(@jroesch): improve
            _ => Err(anyhow::anyhow!("unable to convert datatype from ...")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context() {
        let ctx = Context::cpu(0);
        println!("ctx: {}", ctx);
        let default_ctx = Context::new(DeviceType::CPU, 0);
        assert_eq!(ctx.clone(), default_ctx);
        assert_ne!(ctx, Context::gpu(0));

        let str_ctx = Context::new(DeviceType::GPU, 0);
        assert_eq!(str_ctx.clone(), str_ctx);
        assert_ne!(str_ctx, Context::new(DeviceType::CPU, 0));
    }
}
