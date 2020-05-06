use std::convert::TryFrom;
use std::convert::TryInto;
use std::ffi::CString;
use tvm_sys::{ArgValue, RetValue};
use crate::external_func;

mod object_ptr;

pub use object_ptr::{IsObject, Object, ObjectPtr};

#[derive(Clone)]
pub struct ObjectRef(pub Option<ObjectPtr<Object>>);

impl ObjectRef {
    pub fn null() -> ObjectRef {
        ObjectRef(None)
    }
}

pub trait ToObjectRef {
    fn to_object_ref(&self) -> ObjectRef;
}

impl ToObjectRef for ObjectRef {
    fn to_object_ref(&self) -> ObjectRef {
        self.clone()
    }
}

// impl<T: ToObjectRef> ToObjectRef for &T {
//     fn to_object_ref(&self) -> ObjectRef {
//         (*self).to_object_ref()
//     }
// }

impl TryFrom<RetValue> for ObjectRef {
    type Error = anyhow::Error;

    fn try_from(ret_val: RetValue) -> Result<ObjectRef, Self::Error> {
        let optr = ret_val.try_into()?;
        Ok(ObjectRef(Some(optr)))
    }
}

impl From<ObjectRef> for RetValue {
    fn from(object_ref: ObjectRef) -> RetValue {
        use std::ffi::c_void;
        let object_ptr = &object_ref.0;
        match object_ptr {
            None => RetValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void),
            Some(value) => value.clone().into(),
        }
    }
}

impl<'a> std::convert::TryFrom<ArgValue<'a>> for ObjectRef {
    type Error = anyhow::Error;

    fn try_from(arg_value: ArgValue<'a>) -> Result<ObjectRef, Self::Error> {
        let optr = arg_value.try_into()?;
        Ok(ObjectRef(Some(optr)))
    }
}

impl<'a> std::convert::TryFrom<&ArgValue<'a>> for ObjectRef {
    type Error = anyhow::Error;

    fn try_from(arg_value: &ArgValue<'a>) -> Result<ObjectRef, Self::Error> {
        // TODO(@jroesch): remove the clone
        let value: ArgValue<'a> = arg_value.clone();
        ObjectRef::try_from(value)
    }
}

impl<'a> From<ObjectRef> for ArgValue<'a> {
    fn from(object_ref: ObjectRef) -> ArgValue<'a> {
        use std::ffi::c_void;
        let object_ptr = &object_ref.0;
        match object_ptr {
            None => ArgValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void),
            Some(value) => value.clone().into(),
        }
    }
}

impl<'a> From<&ObjectRef> for ArgValue<'a> {
    fn from(object_ref: &ObjectRef) -> ArgValue<'a> {
        let oref: ObjectRef = object_ref.clone();
        ArgValue::<'a>::from(oref)
    }
}

external_func! {
    fn debug_print(object: ObjectRef) -> CString as "ir.DebugPrinter";
}

external_func! {
    fn as_text(object: ObjectRef) -> CString as "ir.TextPrinter";
}
