#[derive(Debug, Fail)]
#[fail(
    display = "Could not downcast `{}` into `{}`",
    expected_type, actual_type
)]
pub struct ValueDowncastError {
    pub actual_type: String,
    pub expected_type: &'static str,
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
