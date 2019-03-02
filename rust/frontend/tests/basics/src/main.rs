extern crate ndarray as rust_ndarray;
extern crate tvm_frontend as tvm;

use tvm::*;

fn main() {
    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];

    let (ctx, ctx_name) = if cfg!(feature = "cpu") {
        (TVMContext::cpu(0), "cpu")
    } else {
        (TVMContext::gpu(0), "gpu")
    };
    let dtype = TVMType::from("float32");
    let mut arr = NDArray::empty(shape, ctx, dtype);
    arr.copy_from_buffer(data.as_mut_slice());
    let mut ret = NDArray::empty(shape, ctx, dtype);
    let mut fadd = Module::load(&concat!(env!("OUT_DIR"), "/test_add.so")).unwrap();
    if !fadd.enabled(ctx_name) {
        return;
    }
    if cfg!(feature = "gpu") {
        fadd.import_module(Module::load(&concat!(env!("OUT_DIR"), "/test_add.ptx")).unwrap());
    }
    function::Builder::from(&mut fadd)
        .arg(&arr)
        .arg(&arr)
        .set_output(&mut ret)
        .unwrap()
        .invoke()
        .unwrap();

    assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
}
