pub use failure::Error;

#[derive(Debug, Fail)]
#[fail(display = "Cannot convert from an empty array.")]
pub struct EmptyArrayError;

#[derive(Debug, Fail)]
#[fail(display = "Handle `{}` is null.", name)]
pub struct NullHandleError {
    pub name: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Function was not set in `function::Builder`")]
pub struct FunctionNotFoundError;

#[derive(Debug, Fail)]
#[fail(display = "Expected type `{}` but found `{}`", expected, actual)]
pub struct TypeMismatchError {
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Missing NDArray shape.")]
pub struct MissingShapeError;
