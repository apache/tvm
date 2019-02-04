use crate::{
    Module, TVMContext, TVMDeviceType,
    TVMByteArray, NDArray, Result, Function
};
use std::{
    fs,
    path::Path,
    convert::TryInto
};

#[allow(unused)]
pub struct GraphModule {
    module: Module,
    set_input_: Function,
    get_input_: Function,
    get_output_: Function,
    get_num_outputs_: Function,
    run_: Function,
    load_params_: Function
}

impl GraphModule {
    pub fn from_paths<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        graph_json_path: P,
        lib_path: Q,
        params_path: R,
        ctx: TVMContext
    ) -> Result<Self> {
        let graph = fs::read_to_string(&graph_json_path)?;
        let lib = Module::load(&lib_path)?;
        let params: Vec<u8> = fs::read(&params_path)?;
        let mut result = GraphModule::new(&graph[..], &lib, ctx)?;
        result.load_params(TVMByteArray::from(&params));
        Ok(result)
    }

    pub fn new(graph: &str, lib: &Module, ctx: TVMContext) -> Result<Self> {
        let module = packed::create(graph, lib, &ctx.device_type, &ctx.device_id)?;

        let set_input_ = module.get_function("set_input", false).expect("no set_input, invalid GraphModule");
        let get_input_ = module.get_function("get_input", false).expect("no get_input, invalid GraphModule");
        let get_output_ = module.get_function("get_output", false).expect("no get_output, invalid GraphModule");
        let get_num_outputs_ = module.get_function("get_num_outputs", false).expect("no get_num_outputs, invalid GraphModule");
        let run_ = module.get_function("run", false).expect("no run_, invalid GraphModule");
        let load_params_ = module.get_function("load_params", false).expect("no load_params_, invalid GraphModule");

        Ok(GraphModule { module, set_input_, get_input_, get_output_, get_num_outputs_, run_, load_params_ })
    }

    // TODO: should these functions return Result?

    pub fn load_params<B: Into<TVMByteArray>>(&mut self, params: B) {
        let params: TVMByteArray = params.into();
        call_packed!(&self.load_params_, &params).expect("load params failed");
    }

    pub fn set_input(&mut self, name: &str, input: &NDArray) {
        call_packed!(&self.set_input_, name, input).expect("set input failed");
    }

    pub fn get_num_outputs(&mut self) -> usize {
        let num_outputs: i32 = call_packed!(&self.get_num_outputs_,).expect("get num outputs failed")
            .try_into().expect("incorrect num_outputs type");
        num_outputs as usize
    }

    pub fn set_input_by_index(&mut self, index: usize, input: &NDArray) {
        call_packed!(&self.set_input_, &index, input).expect("set input failed");
    }

    pub fn copy_output_to(&mut self, index: usize, output: &NDArray) {
        call_packed!(&self.get_output_, &index, output).expect("get output failed");
    }

    pub fn get_output(&mut self, index: usize) -> NDArray {
        call_packed!(&self.get_output_, &index).expect("get output failed")
            .try_into().expect("incorrect get_output type")
    }

    pub fn run(&mut self) {
        call_packed!(&self.run_,).expect("run failed");
    }

    pub fn apply(&mut self, inputs: &[&NDArray]) -> Vec<NDArray> {
        for (i, input) in inputs.iter().enumerate() {
            self.set_input_by_index(i, input);
        }
        self.run();
        let outputs = self.get_num_outputs();
        (0..outputs).map(|i| self.get_output(i)).collect()
    }
}

wrap_packed_globals! {
    fn tvm.graph_runtime::create(
        graph: &str,
        lib: &Module,
        ty: &TVMDeviceType,
        id: &usize
    ) -> Result<Module>;
}