use crate::ir::array::Array;
use crate::runtime::{external, Function, String as TString};
use crate::runtime::{Object, ObjectPtr, ObjectRef};
use tvm_macros::Object;

type Pass = ObjectRef;

#[repr(C)]
#[derive(Object)]
#[ref_name = "PassInfo"]
#[type_key = "transform.PassInfo"]
pub struct PassInfoNode {
    pub base: Object,
    pub opt_level: i32,
    pub name: TString,
    pub required: Array<TString>,
}

impl PassInfo {
    pub fn new(opt_level: i32, name: String, required: Vec<String>) -> anyhow::Result<PassInfo> {
        let required: Result<_, _> = required
            .into_iter()
            .map(|name| TString::new(name))
            .collect();

        let required = Array::from_vec(required?)?;

        let node = PassInfoNode {
            base: Object::base_object::<PassInfoNode>(),
            opt_level,
            name: TString::new(name).unwrap(),
            required,
        };

        Ok(PassInfo(Some(ObjectPtr::new(node))))
    }
}

external! {
    #[name("relay._transform.MakeFunctionPass")]
    fn create_func_pass(func: Function, pass_info: PassInfo) -> Pass;
}
