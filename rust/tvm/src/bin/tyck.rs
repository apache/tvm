use std::path::PathBuf;

use anyhow::Result;
use structopt::StructOpt;

use tvm::ir::diagnostics::codespan;
use tvm::ir::{self, IRModule};
use tvm::runtime::Error;

#[derive(Debug, StructOpt)]
#[structopt(name = "tyck", about = "Parse and type check a Relay program.")]
struct Opt {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn main() -> Result<()> {
    codespan::init().expect("Rust based diagnostics");
    let opt = Opt::from_args();
    println!("{:?}", &opt);
    let _module = match IRModule::parse_file(opt.input) {
        Err(ir::module::Error::TVM(Error::DiagnosticError(_))) => return Ok(()),
        Err(e) => {
            return Err(e.into());
        }
        Ok(module) => module,
    };

    Ok(())
}
