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

//! Provides [`Device`] and related device queries.
//!
//! Create a new device for device type and device id.
//!
//! # Example
//!
//! ```
//! # use tvm_sys::{DeviceType, Device};
//! let cpu = DeviceType::from("cpu");
//! let dev = Device::new(cpu , 0);
//! let cpu0 = Device::cpu(0);
//! assert_eq!(dev, cpu0);
//! ```
//!
//! Or from a supported device name.
//!
//! ```
//! use tvm_sys::Device;
//! let cpu0 = Device::from("cpu");
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
/// [TVM](https://github.com/apache/tvm).
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
    CUDA = 2,
    CUDAHost = 3,
    OpenCL = 4,
    Vulkan = 7,
    Metal = 8,
    VPI = 9,
    ROCM = 10,
    ExtDev = 12,
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
                DeviceType::CUDA => "cuda",
                DeviceType::CUDAHost => "cuda_host",
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
            "cuda" => DeviceType::CUDA,
            "nvptx" => DeviceType::CUDA,
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
pub struct Device {
    pub device_type: DeviceType,
    pub device_id: usize,
}

impl Device {
    pub fn new(device_type: DeviceType, device_id: usize) -> Device {
        Device {
            device_type,
            device_id,
        }
    }
}

impl<'a> From<&'a Device> for DLDevice {
    fn from(dev: &'a Device) -> Self {
        Self {
            device_type: dev.device_type.into(),
            device_id: dev.device_id as i32,
        }
    }
}

impl Default for Device {
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

macro_rules! impl_tvm_device {
    ( $( $dev_type:ident : [ $( $dev_name:ident ),+ ] ),+ ) => {
        /// Creates a Device from a string (e.g., "cpu", "cuda", "ext_dev")
        impl FromStr for Device {
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

        impl Device {
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

impl_tvm_device!(
    DLDeviceType_kDLCPU: [cpu, llvm, stackvm],
    DLDeviceType_kDLCUDA: [cuda, nvptx],
    DLDeviceType_kDLOpenCL: [cl],
    DLDeviceType_kDLMetal: [metal],
    DLDeviceType_kDLVPI: [vpi],
    DLDeviceType_kDLROCM: [rocm],
    DLDeviceType_kDLExtDev: [ext_dev]
);

impl<'a> From<&'a str> for Device {
    fn from(target: &str) -> Self {
        Device::new(DeviceType::from(target), 0)
    }
}

impl From<ffi::DLDevice> for Device {
    fn from(dev: ffi::DLDevice) -> Self {
        Device {
            device_type: DeviceType::from(dev.device_type),
            device_id: dev.device_id as usize,
        }
    }
}

impl From<Device> for ffi::DLDevice {
    fn from(dev: Device) -> Self {
        ffi::DLDevice {
            device_type: dev.device_type.into(),
            device_id: dev.device_id as i32,
        }
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.device_type, self.device_id)
    }
}

impl<'a> From<&'a Device> for ArgValue<'a> {
    fn from(dev: &'a Device) -> Self {
        DLDevice::from(dev).into()
    }
}

impl<'a> From<Device> for ArgValue<'a> {
    fn from(dev: Device) -> Self {
        DLDevice::from(dev).into()
    }
}

impl From<Device> for RetValue {
    fn from(ret_value: Device) -> RetValue {
        RetValue::Device(ret_value.into())
    }
}

impl TryFrom<RetValue> for Device {
    type Error = anyhow::Error;
    fn try_from(ret_value: RetValue) -> anyhow::Result<Device> {
        match ret_value {
            RetValue::Device(dt) => Ok(dt.into()),
            // TODO(@jroesch): improve
            _ => Err(anyhow::anyhow!("unable to convert datatype from ...")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device() {
        let dev = Device::cpu(0);
        println!("device: {}", dev);
        let default_dev = Device::new(DeviceType::CPU, 0);
        assert_eq!(dev.clone(), default_dev);
        assert_ne!(dev, Device::cuda(0));

        let str_dev = Device::new(DeviceType::CUDA, 0);
        assert_eq!(str_dev.clone(), str_dev);
        assert_ne!(str_dev, Device::new(DeviceType::CPU, 0));
    }
}
