use crate::runtime::{external, Object, ObjectRef};
use crate::runtime::{string::String as TVMString};
use crate::runtime::map::Map;

use super::expr::GlobalVar;
use super::function::BaseFunc;

use std::io::Result;
use std::path::Path;

use tvm_macros::Object;

// TODO(@jroesch): define type
type TypeData = ObjectRef;
type GlobalTypeVar = ObjectRef;

#[repr(C)]
#[derive(Object)]
#[ref_name = "IRModule"]
#[type_key = "IRModule"]
pub struct IRModuleNode {
    pub base: Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub type_definitions: Map<GlobalTypeVar, TypeData>,
}


external! {
    #[name("parser.ParseModule")]
    fn parse_module(file_name: TVMString, source: TVMString) -> IRModule;
    #[name("parser.ParseExpr")]
    fn parse_expression(file_name: TVMString, source: TVMString) -> IRModule;
}

impl IRModule {
    pub fn parse<N, S>(file_name: N, source: S) -> IRModule
    where N: Into<TVMString>, S: Into<TVMString> {
        parse_module(file_name.into(), source.into())
            .expect("failed to call parser")
    }

    pub fn parse_file<P: 'static + AsRef<Path>>(file_path: P) -> Result<IRModule> {
        let file_path = file_path.as_ref();
        let file_path_as_str = file_path.to_str().unwrap().to_string();
        let source = std::fs::read_to_string(file_path)?;
        let module = IRModule::parse(file_path_as_str, source);
        Ok(module)
    }
}
