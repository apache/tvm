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

// TODO(@jroesch): define type
type TypeData = ObjectRef;
type GlobalTypeVar = ObjectRef;

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
    // Module methods
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
}

// TVM_REGISTER_GLOBAL("ir.Module_GetGlobalTypeVars")
//     .set_body_method<IRModule>(&IRModuleNode::GetGlobalTypeVars);

// TVM_REGISTER_GLOBAL("ir.Module_ContainGlobalVar")
//     .set_body_method<IRModule>(&IRModuleNode::ContainGlobalVar);

// TVM_REGISTER_GLOBAL("ir.Module_GetGlobalTypeVar")
//     .set_body_method<IRModule>(&IRModuleNode::GetGlobalTypeVar);

// TVM_REGISTER_GLOBAL("ir.Module_LookupDef").set_body_typed([](IRModule mod, GlobalTypeVar var) {
//   return mod->LookupTypeDef(var);
// });

// TVM_REGISTER_GLOBAL("ir.Module_LookupDef_str").set_body_typed([](IRModule mod, String var) {
//   return mod->LookupTypeDef(var);
// });

// TVM_REGISTER_GLOBAL("ir.Module_LookupTag").set_body_typed([](IRModule mod, int32_t tag) {
//   return mod->LookupTag(tag);
// });

// TVM_REGISTER_GLOBAL("ir.Module_FromExpr")
//     .set_body_typed([](RelayExpr e, tvm::Map<GlobalVar, BaseFunc> funcs,
//                        tvm::Map<GlobalTypeVar, TypeData> type_defs) {
//       return IRModule::FromExpr(e, funcs, type_defs);
//     });

// TVM_REGISTER_GLOBAL("ir.Module_Update").set_body_typed([](IRModule mod, IRModule from) {
//   mod->Update(from);
// });

// TVM_REGISTER_GLOBAL("ir.Module_UpdateFunction")
//     .set_body_typed([](IRModule mod, GlobalVar gv, BaseFunc func) { mod->Update(gv, func); });

// TVM_REGISTER_GLOBAL("ir.Module_Import").set_body_typed([](IRModule mod, String path) {
//   mod->Import(path);
// });

// TVM_REGISTER_GLOBAL("ir.Module_ImportFromStd").set_body_typed([](IRModule mod, String path) {
//   mod->ImportFromStd(path);
// });

// TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
//     .set_dispatch<IRModuleNode>([](const ObjectRef& ref, ReprPrinter* p) {
//       auto* node = static_cast<const IRModuleNode*>(ref.get());
//       p->stream << "IRModuleNode( " << node->functions << ")";
//     });

impl IRModule {
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
}
