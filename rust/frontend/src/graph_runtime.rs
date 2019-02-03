use crate::{ Module, TVMContext, TVMDeviceType, NDArray, Result, ErrorKind };
use std::fs;
use std::path::Path;

pub struct RuntimeModule {
    module: Module
}

impl RuntimeModule {
    pub fn from_paths<P: AsRef<Path>, Q: AsRef<Path>>(graph_json_path: P, lib_path: Q, ctx: TVMContext) -> Result<Self> {
        let graph = fs::read_to_string(&graph_json_path)?;
        let lib = Module::load(&lib_path)?;
        RuntimeModule::new(&graph[..], &lib, ctx)
    }

    pub fn new(graph: &str, lib: &Module, ctx: TVMContext) -> Result<Self> {
        let module = packed::create(graph, lib, &ctx.device_type, &ctx.device_id)?;
        Ok(RuntimeModule { module })
    }
}

wrap_packed_globals! {
    fn tvm.graph_runtime::create(graph: &str, lib: &Module, ty: &TVMDeviceType, id: &usize) -> Result<Module>;
}