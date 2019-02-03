use crate::ffi::*;

impl TVMType {
    fn new(type_code: u8, bits: u8, lanes: u16) -> Self {
        Self {
            code: type_code,
            bits,
            lanes,
        }
    }
}

/// Implements TVMType conversion from `&str` of general format `{dtype}{bits}x{lanes}`
/// such as "int32", "float32" or with lane "float32x1".
impl<'a> From<&'a str> for TVMType {
    fn from(type_str: &'a str) -> Self {
        if type_str == "bool" {
            return TVMType::new(1, 1, 1);
        }

        let mut type_lanes = type_str.split("x");
        let typ = type_lanes.next().expect("Missing dtype");
        let lanes = type_lanes
            .next()
            .map(|l| u16::from_str_radix(l, 10).expect(&format!("Bad dtype lanes: {}", l)))
            .unwrap_or(1);
        let (type_name, bits) = match typ.find(char::is_numeric) {
            Some(idx) => {
                let (name, bits_str) = typ.split_at(idx);
                (
                    name,
                    u8::from_str_radix(bits_str, 10)
                        .expect(&format!("Bad dtype bits: {}", bits_str)),
                )
            }
            None => (typ, 32),
        };

        let type_code = match type_name {
            "int" => 0,
            "uint" => 1,
            "float" => 2,
            "handle" => 3,
            _ => unimplemented!(),
        };

        TVMType::new(type_code, bits, lanes)
    }
}

impl std::fmt::Display for TVMType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.bits == 1 && self.lanes == 1 {
            return write!(f, "bool");
        }
        let mut type_str = match self.code {
            0 => "int",
            1 => "uint",
            2 => "float",
            4 => "handle",
            _ => "unknown",
        }
        .to_string();

        type_str += &self.bits.to_string();
        if self.lanes > 1 {
            type_str += &format!("x{}", self.lanes);
        }
        f.write_str(&type_str)
    }
}

macro_rules! impl_tvm_val_from_pod {
    ($field:ident, $ty:ty) => {
        impl From<$ty> for TVMValue {
            fn from(val: $ty) -> Self {
                TVMValue { $field: val }
            }
        }
    };
}

impl_tvm_val_from_pod!(v_type, TVMType);
impl_tvm_val_from_pod!(v_ctx, TVMContext);

impl From<DLDeviceType> for TVMValue {
    fn from(dev: DLDeviceType) -> Self {
        TVMValue {
            v_int64: dev as i64,
        }
    }
}
