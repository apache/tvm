use std::convert::TryFrom;
use std::convert::TryInto;
use std::ffi::CString;
use tvm_sys::{ArgValue, RetValue};

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
