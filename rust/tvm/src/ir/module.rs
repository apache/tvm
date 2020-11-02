/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use std::iter::FromIterator;
use std::path::Path;

use thiserror::Error;
use tvm_macros::Object;

use crate::runtime::array::Array;
use crate::runtime::function::Result;
use crate::runtime::map::Map;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, Object, ObjectRef};

use super::expr::GlobalVar;
use super::function::BaseFunc;
use super::source_map::SourceMap;
use super::{ty::GlobalTypeVar, relay};

// TODO(@jroesch): define type
type TypeData = ObjectRef;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("{0}")]
    TVM(#[from] crate::runtime::Error),
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "IRModule"]
#[type_key = "IRModule"]
pub struct IRModuleNode {
    pub base: Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub type_definitions: Map<GlobalTypeVar, TypeData>,
    pub source_map: SourceMap,
    // TODO(@jroesch): this is missing some fields
}

external! {
    // Parser functions
    #[name("parser.ParseModule")]
    fn parse_module(file_name: TVMString, source: TVMString) -> IRModule;
    #[name("parser.ParseExpr")]
    fn parse_expression(file_name: TVMString, source: TVMString) -> IRModule;
    #[name("ir.IRModule")]
    fn module_new(funcs: Map<GlobalVar, BaseFunc>, types: Map<GlobalTypeVar, TypeData>) -> IRModule;
    // Module methods
    #[name("ir.Module_Add")]
    fn module_add(module: IRModule, type_name: GlobalVar, expr: BaseFunc, update: bool) -> IRModule;
    #[name("ir.Module_AddDef")]
    fn module_add_def(module: IRModule, type_name: GlobalTypeVar, type_data: TypeData, update: bool) -> ();
    #[name("ir.Module_GetGlobalVar")]
    fn module_get_global_var(module: IRModule, name: TVMString) -> GlobalVar;
    #[name("ir.Module_GetGlobalVars")]
    fn module_get_global_vars(module: IRModule) -> Array<GlobalVar>;
    #[name("ir.Module_Lookup")]
    fn module_lookup(module: IRModule, var: GlobalVar) -> BaseFunc;
    #[name("ir.Module_Lookup_str")]
    fn module_lookup_str(module: IRModule, name: TVMString) -> BaseFunc;
    #[name("ir.Module_GetGlobalTypeVars")]
    fn module_get_global_type_vars(module: IRModule) -> Array<GlobalTypeVar>;
    #[name("ir.Module_ContainGlobalVar")]
    fn module_contains_global_var(module: IRModule, name: TVMString) -> bool;
    #[name("ir.Module_ContainGlobalTypeVar")]
    fn module_contains_global_type_var(module: IRModule, name: TVMString) -> bool;
    #[name("ir.Module_LookupDef")]
    fn module_lookup_def(module: IRModule, global: GlobalTypeVar) -> TypeData;
    #[name("ir.Module_LookupDef_str")]
    fn module_lookup_def_str(module: IRModule, global: GlobalTypeVar) -> TypeData;
    #[name("ir.Module_LookupTag")]
    fn module_lookup_tag(module: IRModule, tag: i32) -> relay::Constructor;
    #[name("ir.Module_FromExpr")]
    fn module_from_expr(expr: relay::Expr, funcs: Map<GlobalVar, BaseFunc>, types: Map<GlobalTypeVar, TypeData>) -> IRModule;
    #[name("ir.Module_Import")]
    fn module_import(module: IRModule, path: TVMString);
    #[name("ir.Module_ImportFromStd")]
    fn module_import_from_std(module: IRModule, path: TVMString);
}

// Note: we don't expose update here as update is going to be removed.

impl IRModule {
    pub fn new<F, T>(funcs: F, types: T) -> Result<IRModule>
    where F: IntoIterator<Item=(GlobalVar, BaseFunc)>, T: IntoIterator<Item=(GlobalTypeVar, TypeData)> {
        module_new(Map::from_iter(funcs), Map::from_iter(types))
    }

    pub fn parse<N, S>(file_name: N, source: S) -> Result<IRModule>
    where
        N: Into<TVMString>,
        S: Into<TVMString>,
    {
        parse_module(file_name.into(), source.into())
    }

    pub fn parse_file<P: 'static + AsRef<Path>>(
        file_path: P,
    ) -> std::result::Result<IRModule, Error> {
        let file_path = file_path.as_ref();
        let file_path_as_str = file_path.to_str().unwrap().to_string();
        let source = std::fs::read_to_string(file_path)?;
        let module = IRModule::parse(file_path_as_str, source)?;
        Ok(module)
    }

    pub fn add(
        &mut self,
        var: GlobalVar,
        func: BaseFunc) -> Result<IRModule> {
            module_add(self.clone(), var, func, true)
        }

    pub fn add_def(
        &mut self,
        type_name: GlobalTypeVar,
        type_data: TypeData,
        update: bool,
    ) -> Result<()> {
        module_add_def(self.clone(), type_name, type_data, update)
    }

    pub fn get_global_var(&self, name: TVMString) -> Result<GlobalVar> {
        module_get_global_var(self.clone(), name)
    }

    pub fn get_global_vars(&self) -> Result<Array<GlobalVar>> {
        module_get_global_vars(self.clone())
    }

    pub fn lookup(&self, var: GlobalVar) -> Result<BaseFunc> {
        module_lookup(self.clone(), var)
    }

    pub fn lookup_str<S>(&self, name: S) -> Result<BaseFunc>
    where
        S: Into<TVMString>,
    {
        module_lookup_str(self.clone(), name.into())
    }

    pub fn get_global_type_vars(&self) -> Result<Array<GlobalTypeVar>> {
        module_get_global_type_vars(self.clone())
    }

    pub fn contains_global_var<S: Into<TVMString>>(&self, name: S) -> Result<bool> {
        module_contains_global_var(self.clone(), name.into())
    }

    pub fn contains_global_type_var<S: Into<TVMString>>(&self, name: S) -> Result<bool> {
        module_contains_global_type_var(self.clone(), name.into())
    }

    pub fn lookup_def(&self, global: GlobalTypeVar) -> Result<TypeData> {
        module_lookup_def(self.clone(), global)
    }

    pub fn lookup_def_str(&self, global: GlobalTypeVar) -> Result<TypeData> {
        module_lookup_def_str(self.clone(), global)
    }

    pub fn lookup_tag(&self, tag: i32) -> Result<relay::Constructor> {
        module_lookup_tag(self.clone(), tag)
    }

    pub fn from_expr(expr: relay::Expr, funcs: Map<GlobalVar, BaseFunc>, types: Map<GlobalTypeVar, TypeData>) -> Result<IRModule> {
        module_from_expr(expr, funcs, types)
    }

    pub fn import<S: Into<TVMString>>(&mut self, path: S) -> Result<()> {
        module_import(self.clone(), path.into())
    }

    pub fn import_from_std<S: Into<TVMString>>(&mut self, path: S) -> Result<()> {
        module_import_from_std(self.clone(), path.into())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use super::relay::*;
    use super::*;
    use super::super::span::Span;
    use tvm_rt::IsObjectRef;

    #[test]
    fn test_module_add() -> anyhow::Result<()> {
        let funcs = HashMap::<GlobalVar, BaseFunc>::new();
        let types =  HashMap::<GlobalTypeVar, TypeData>::new();
        let mut module = IRModule::new(funcs, types)?;
        let x = Var::static_tensor("x".into(), vec![1, 1], DataType::float32());
        let params = Array::from_vec(vec![x.clone()])?;
        let func = relay::Function::simple(params, x.upcast()).upcast();
        let module = module.add(GlobalVar::new("foo".into(), Span::null()), func)?;
        // let lfunc = module.lookup_str("foo")?;
        // let lfunc = lfunc.downcast::<relay::Function>()?;
        // assert_eq!(lfunc.params.len(), 1);
        Ok(())
    }

    #[test]
    fn test_module_add_def() {

    }

    #[test]
    fn test_get_global_var() {

    }

    #[test]
    fn test_get_global_vars() {

    }

    #[test]
    fn test_lookup() {

    }


    // pub fn get_global_type_vars(&self) -> Result<Array<GlobalTypeVar>> {
    //     module_get_global_type_vars(self.clone())
    // }

    // pub fn contains_global_var<S: Into<TVMString>>(&self, name: S) -> Result<bool> {
    //     module_contains_global_var(self.clone(), name.into())
    // }

    // pub fn contains_global_type_var<S: Into<TVMString>>(&self, name: S) -> Result<bool> {
    //     module_contains_global_type_var(self.clone(), name.into())
    // }

    #[test]
    fn test_lookup_def() {

    }
    // pub fn lookup_def(&self, global: GlobalTypeVar) -> Result<TypeData> {
    //     module_lookup_def(self.clone(), global)
    // }

    // pub fn lookup_def_str(&self, global: GlobalTypeVar) -> Result<TypeData> {
    //     module_lookup_def_str(self.clone(), global)
    // }

    // pub fn lookup_tag(&self, tag: i32) -> Result<relay::Constructor> {
    //     module_lookup_tag(self.clone(), tag)
    // }

    // pub fn from_expr(expr: relay::Expr, funcs: Map<GlobalVar, BaseFunc>, types: Map<GlobalTypeVar, TypeData>) -> Result<IRModule> {
    //     module_from_expr(expr, funcs, types)
    // }


    // pub fn import<S: Into<TVMString>>(&mut self, path: S) -> Result<()> {
    //     module_import(self.clone(), path.into())
    // }


    // pub fn import_from_std<S: Into<TVMString>>(&mut self, path: S) -> Result<()> {
    //     module_import_from_std(self.clone(), path.into())
    // }
}
