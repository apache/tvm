use std::fmt;

static TYPE_CODE_STRS: [&str; 15] = [
    "int",
    "uint",
    "float",
    "handle",
    "null",
    "TVMType",
    "TVMContext",
    "ArrayHandle",
    "NodeHandle",
    "ModuleHandle",
    "FuncHandle",
    "str",
    "bytes",
    "NDArrayContainer",
    "ExtBegin",
];

#[derive(Debug, Fail)]
pub struct ValueDowncastError {
    actual_type_code: i64,
    expected_type_code: i64,
}

impl ValueDowncastError {
    pub fn new(actual_type_code: i64, expected_type_code: i64) -> Self {
        Self {
            actual_type_code,
            expected_type_code,
        }
    }
}

impl fmt::Display for ValueDowncastError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Could not downcast TVMValue: expected `{}` but was {}",
            TYPE_CODE_STRS[self.actual_type_code as usize],
            TYPE_CODE_STRS[self.expected_type_code as usize]
        )
    }
}

#[derive(Debug, Fail)]
#[fail(display = "Function call `{}` returned error: {}", context, message)]
pub struct FuncCallError {
    context: String,
    message: String,
}

impl FuncCallError {
    pub fn get_with_context(context: String) -> Self {
        Self {
            context,
            message: unsafe { std::ffi::CStr::from_ptr(crate::ffi::TVMGetLastError()) }
                .to_str()
                .expect("double fault")
                .to_owned(),
        }
    }
}

// error_chain! {
//     errors {
//         TryFromTVMRetValueError(expected_type: String, actual_type_code: i64) {
//             description("mismatched types while downcasting TVMRetValue")
//             display("invalid downcast: expected `{}` but was `{}`",
//                     expected_type, type_code_to_string(actual_type_code))
//         }
//     }
//     foreign_links {
//         IntoString(std::ffi::IntoStringError);
//         ParseInt(std::num::ParseIntError);
//         Utf8(std::str::Utf8Error);
//     }
// }
