use std::convert::TryFrom;
use std::convert::TryInto;
use std::ffi::CString;
use tvm_sys::{TVMArgValue, TVMRetValue};

mod object_ptr;

pub use object_ptr::{Object, IsObject, ObjectPtr};

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

impl TryFrom<TVMRetValue> for ObjectRef {
    type Error = anyhow::Error;

    fn try_from(ret_val: TVMRetValue) -> Result<ObjectRef, Self::Error> {
        let optr = ret_val.try_into()?;
        Ok(ObjectRef(Some(optr)))
    }
}

impl From<ObjectRef> for TVMRetValue {
    fn from(object_ref: ObjectRef) -> TVMRetValue {
        use std::ffi::c_void;
        let object_ptr = &object_ref.0;
        match object_ptr {
            None => {
                TVMRetValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void)
            }
            Some(value) => value.clone().into()
        }
    }
}

impl<'a> std::convert::TryFrom<TVMArgValue<'a>> for ObjectRef {
    type Error = anyhow::Error;

    fn try_from(arg_value: TVMArgValue<'a>) -> Result<ObjectRef, Self::Error> {
        let optr = arg_value.try_into()?;
        Ok(ObjectRef(Some(optr)))
    }
}

impl<'a> std::convert::TryFrom<&TVMArgValue<'a>> for ObjectRef {
    type Error = anyhow::Error;

    fn try_from(arg_value: &TVMArgValue<'a>) -> Result<ObjectRef, Self::Error> {
        // TODO(@jroesch): remove the clone
        let value: TVMArgValue<'a> = arg_value.clone();
        ObjectRef::try_from(value)
    }
}

impl<'a> From<ObjectRef> for TVMArgValue<'a> {

    fn from(object_ref: ObjectRef) -> TVMArgValue<'a> {
        use std::ffi::c_void;
        let object_ptr = &object_ref.0;
        match object_ptr {
            None => {
                TVMArgValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void)
            }
            Some(value) => value.clone().into()
        }
    }
}

impl<'a> From<&ObjectRef> for TVMArgValue<'a> {
    fn from(object_ref: &ObjectRef) -> TVMArgValue<'a> {
        let oref: ObjectRef = object_ref.clone();
        TVMArgValue::<'a>::from(oref)
    }
}

#[macro_export]
macro_rules! external_func {
    (fn $name:ident ( $($arg:ident : $ty:ty),* ) -> $ret_type:ty as $ext_name:literal;) => {
        ::paste::item! {
            #[allow(non_upper_case_globals)]
            static [<global_ $name>]: ::once_cell::sync::Lazy<&'static $crate::Function> =
            ::once_cell::sync::Lazy::new(|| {
                $crate::Function::get($ext_name)
                .expect(concat!("unable to load external function", stringify!($ext_name), "from TVM registry."))
            });
        }

        pub fn $name($($arg : $ty),*) -> Result<$ret_type, anyhow::Error> {
            use std::convert::TryInto;
            let func_ref: &$crate::Function = ::paste::expr! { &*[<global_ $name>] };
            let res = $crate::call_packed!(func_ref,$($arg),*)?;
            let res = res.try_into()?;
            Ok(res)
        }
    }
}

external_func! {
    fn debug_print(object: &ObjectRef) -> CString as "ir.DebugPrinter";
}

external_func! {
    fn as_text(object: &ObjectRef) -> CString as "ir.TextPrinter";
}
