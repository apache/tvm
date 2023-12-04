use crate::object::Object;
use tvm_macros::Object;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BoxBool"]
#[type_key = "runtime.BoxBool"]
pub struct BoxBoolNode {
    base: Object,
    value: bool,
}

impl From<bool> for BoxBool {
    fn from(value: bool) -> Self {
        _box_bool(value as i64).expect("Failed to box boolean for FFI")
    }
}

impl Into<bool> for BoxBool {
    fn into(self) -> bool {
        self.value
    }
}

crate::external! {
    #[name("runtime.BoxBool")]
    fn _box_bool(value: i64) -> BoxBool;

    #[name("runtime.UnBoxBool")]
    fn _unbox_bool(boxed: BoxBool) -> i64;
}
