extern crate ndarray;
#[macro_use]
extern crate tvm;

use ndarray::Array;
use tvm::{
  ffi::runtime::DLTensor,
  runtime::{Module, SystemLibModule},
};

fn main() {
  let syslib = SystemLibModule::default();
  let add = syslib
    .get_function("default_function")
    .expect("main function not found");
  let mut a = Array::from_vec(vec![1f32, 2., 3., 4.]);
  let mut b = Array::from_vec(vec![1f32, 0., 1., 0.]);
  let mut c = Array::from_vec(vec![0f32; 4]);
  let e = Array::from_vec(vec![2f32, 2., 4., 4.]);
  let mut a_dl: DLTensor = (&mut a).into();
  let mut b_dl: DLTensor = (&mut b).into();
  let mut c_dl: DLTensor = (&mut c).into();
  call_packed!(add, &mut a_dl, &mut b_dl, &mut c_dl);
  assert!(c.all_close(&e, 1e-8f32));
}
