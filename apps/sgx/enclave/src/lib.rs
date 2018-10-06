#![feature(try_from)]

#[macro_use]
extern crate lazy_static;
extern crate tvm;

use std::{convert::TryFrom, sync::Mutex};

use tvm::runtime::{sgx, Graph, GraphExecutor, SystemLibModule, TVMArgValue, TVMRetValue};

lazy_static! {
  static ref SYSLIB: SystemLibModule = { SystemLibModule::default() };
  static ref MODEL: Mutex<GraphExecutor<'static, 'static>> = {
    let _params = include_bytes!(concat!("../", env!("BUILD_DIR"), "/params.bin"));
    let graph_json = include_str!(concat!("../", env!("BUILD_DIR"), "/graph.json"));

    let graph = Graph::try_from(graph_json).unwrap();
    Mutex::new(GraphExecutor::new(graph, &*SYSLIB).unwrap())
  };
}

fn ecall_init(_args: &[TVMArgValue]) -> TVMRetValue {
  lazy_static::initialize(&MODEL);
  TVMRetValue::from(0)
}

fn ecall_main(_args: &[TVMArgValue]) -> TVMRetValue {
  let model = MODEL.lock().unwrap();
  // model.set_input("data", args[0]);
  model.run();
  sgx::shutdown();
  // model.get_output(0).into()
  TVMRetValue::from(42)
}

pub mod ecalls {
  //! todo: generate this using proc_macros

  use super::*;

  use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
    slice,
  };

  use tvm::{
    ffi::runtime::{TVMRetValueHandle, TVMValue},
    runtime::{
      sgx::{run_worker, SgxStatus},
      PackedFunc,
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

  lazy_static! {
    static ref ECALL_FUNCS: Vec<PackedFunc> = {
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
        )).expect(&format!("Error registering `{}`", ecall));
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
        .collect::<Vec<TVMArgValue>>()
    };
    let (rv, tc) = ECALL_FUNCS[func_id as usize](&args).into_tvm_value();
    unsafe {
      *ret_val = rv;
      *ret_type_code = tc;
    }
  }
}
