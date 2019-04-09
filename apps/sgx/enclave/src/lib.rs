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
#![feature(try_from)]

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate tvm;

use std::{
  convert::{TryFrom, TryInto},
  sync::Mutex,
};

use tvm::{
  ffi::runtime::DLTensor,
  runtime::{
    load_param_dict, sgx, Graph, GraphExecutor, SystemLibModule, TVMArgValue, TVMRetValue, Tensor,
  },
};

lazy_static! {
  static ref SYSLIB: SystemLibModule = { SystemLibModule::default() };
  static ref MODEL: Mutex<GraphExecutor<'static, 'static>> = {
    let graph_json = include_str!(concat!("../", env!("BUILD_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!("../", env!("BUILD_DIR"), "/params.bin"));
    let params = load_param_dict(params_bytes).unwrap();

    let graph = Graph::try_from(graph_json).unwrap();
    let mut exec = GraphExecutor::new(graph, &*SYSLIB).unwrap();
    exec.load_params(params);
    Mutex::new(exec)
  };
}

fn ecall_init(_args: &[TVMArgValue]) -> TVMRetValue {
  lazy_static::initialize(&MODEL);
  TVMRetValue::from(0)
}

fn ecall_main(args: &[TVMArgValue<'static>]) -> TVMRetValue {
  let mut model = MODEL.lock().unwrap();
  let inp = args[0].try_into().unwrap();
  let mut out: Tensor = args[1].try_into().unwrap();
  model.set_input("data", inp);
  model.run();
  sgx::shutdown();
  out.copy(model.get_output(0).unwrap());
  TVMRetValue::from(1)
}

pub mod ecalls {
  //! todo: generate this using proc_macros

  use super::*;

  use std::{
    ffi::CString,
    mem,
    os::raw::{c_char, c_int, c_void},
    slice,
  };

  use tvm::{
    ffi::runtime::{TVMRetValueHandle, TVMValue},
    runtime::{
      sgx::{ocall_packed_func, run_worker, SgxStatus},
      DataType, PackedFunc,
    },
  };

  macro_rules! tvm_ocall {
    ($func: expr) => {
      match $func {
        0 => Ok(()),
        err => Err(err),
      }
    };
  }

  const ECALLS: &'static [&'static str] = &["__tvm_run_worker__", "__tvm_main__", "init"];

  pub type EcallPackedFunc = Box<Fn(&[TVMArgValue<'static>]) -> TVMRetValue + Send + Sync>;

  lazy_static! {
    static ref ECALL_FUNCS: Vec<EcallPackedFunc> = {
      vec![
        Box::new(run_worker),
        Box::new(ecall_main),
        Box::new(ecall_init),
      ]
    };
  }

  extern "C" {
    fn __tvm_module_startup() -> ();
    fn tvm_ocall_register_export(name: *const c_char, func_id: c_int) -> SgxStatus;
  }

  #[no_mangle]
  pub extern "C" fn tvm_ecall_init(_ret: TVMRetValueHandle) {
    unsafe {
      __tvm_module_startup();

      ECALLS.into_iter().enumerate().for_each(|(i, ecall)| {
        tvm_ocall!(tvm_ocall_register_export(
          CString::new(*ecall).unwrap().as_ptr(),
          i as i32
        ))
        .expect(&format!("Error registering `{}`", ecall));
      });
    }
  }

  #[no_mangle]
  pub extern "C" fn tvm_ecall_packed_func(
    func_id: c_int,
    arg_values: *const TVMValue,
    type_codes: *const c_int,
    num_args: c_int,
    ret_val: *mut TVMValue,
    ret_type_code: *mut i64,
  ) {
    let args = unsafe {
      let values = slice::from_raw_parts(arg_values, num_args as usize);
      let type_codes = slice::from_raw_parts(type_codes, num_args as usize);
      values
        .into_iter()
        .zip(type_codes.into_iter())
        .map(|(v, t)| TVMArgValue::new(*v, *t as i64))
        .collect::<Vec<TVMArgValue<'static>>>()
    };
    let (rv, tc) = ECALL_FUNCS[func_id as usize](&args).into_tvm_value();
    unsafe {
      *ret_val = rv;
      *ret_type_code = tc;
    }
  }
}
