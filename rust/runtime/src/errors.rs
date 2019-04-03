#[derive(Debug, Fail)]
pub enum GraphFormatError {
    #[fail(display = "Could not parse graph json")]
    Parse(#[fail(cause)] failure::Error),
    #[fail(display = "Could not parse graph params")]
    Params,
    #[fail(display = "{} is missing attr: {}", 0, 1)]
    MissingAttr(String, String),
    #[fail(display = "Missing field: {}", 0)]
    MissingField(&'static str),
    #[fail(display = "Invalid DLType: {}", 0)]
    InvalidDLType(String),
}

#[derive(Debug, Fail)]
#[fail(display = "SGX error: 0x{:x}", code)]
pub struct SgxError {
    pub code: u32,
}
