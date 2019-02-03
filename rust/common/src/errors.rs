//! Error types for `TVMArgValue` and `TVMRetValue` conversions.

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
fn type_code_to_string(type_code: &i64) -> String {
    TYPE_CODE_STRS[*type_code as usize].to_string()
}

error_chain! {
    errors {
        // TryFromTVMArgValueError(expected: String, actual: String) {
        //       description("mismatched types while converting from TVMArgValue")
        //       display("expected `{}` but given `{}`", expected, actual)
        // }
        //
        // TryFromTVMRetValueError(expected: String, actual: String) {
        //       description("mismatched types while downcasting TVMRetValue")
        //       display("invalid downcast: expected `{}` but given `{}`", expected, actual)
        // }

        TryFromTVMRetValueError(expected_type: String, actual_type_code: i64) {
            description("mismatched types while downcasting TVMRetValue")
            display("invalid downcast: expected `{}` but was `{}`",
                    expected_type, type_code_to_string(actual_type_code))
        }
    }
}
