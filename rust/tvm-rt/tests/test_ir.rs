extern crate tvm_frontend as tvm;
use tvm::{Function, DLDataType, call_packed};
use tvm::ir::IntImmNode;
use tvm::runtime::{Object, ObjectRef, ObjectPtr, debug_print};
use tvm::runtime::String as TString;
use tvm_sys::TVMRetValue;
use std::str::FromStr;
use std::convert::TryInto;



#[test]
fn test_new_object() -> anyhow::Result<()> {
    let object = Object::base_object::<Object>();
    let ptr = ObjectPtr::new(object);
    assert_eq!(ptr.count(), 1);
    Ok(())
}

#[test]
fn test_new_string() -> anyhow::Result<()> {
    let string = TString::new("hello world!".to_string())?;
    Ok(())
}

#[test]
fn test_obj_build() -> anyhow::Result<()> {
    let int_imm = Function::get("ir.IntImm")
        .expect("Stable TVM API not found.");

    let dt = DLDataType::from_str("int32")
        .expect("Known datatype doesn't convert.");

    let ret_val: ObjectRef = call_packed!(int_imm, dt, 1337).expect("foo").try_into().unwrap();

    debug_print(&ret_val);

    Ok(())
}
