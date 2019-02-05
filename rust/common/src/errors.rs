//! Error types for `TVMArgValue` and `TVMRetValue` conversions.

error_chain! {
    errors {
        TryFromTVMArgValueError(expected: String, actual: String) {
              description("mismatched types while converting from TVMArgValue")
              display("expected `{}` but given `{}`", expected, actual)
        }

        TryFromTVMRetValueError(expected: String, actual: String) {
              description("mismatched types while downcasting TVMRetValue")
              display("invalid downcast: expected `{}` but given `{}`", expected, actual)
        }
    }
}
