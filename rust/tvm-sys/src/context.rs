
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

use crate::ffi::*;
use std::str::FromStr;
use thiserror::Error;

 #[derive(Debug, Clone, Copy, PartialEq)]
pub struct Context {
    pub device_type: usize,
    pub device_id: usize,
}

impl<'a> From<&'a Context> for DLContext {
    fn from(ctx: &'a Context) -> Self {
        Self {
            device_type: ctx.device_type as _,
            device_id: ctx.device_id as i32,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            device_type: DLDeviceType_kDLCPU as usize,
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
                         $( $(  stringify!($dev_name)  )|+ => $dev_type as usize),+,
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
                            device_type: $dev_type as usize,
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
