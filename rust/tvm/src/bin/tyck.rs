use std::path::PathBuf;

use anyhow::Result;
use structopt::StructOpt;

use tvm::ir::diagnostics::codespan;
use tvm::ir::IRModule;

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
    let module = IRModule::parse_file(opt.input);

    // for (k, v) in module.functions {
    //     println!("Function name: {:?}", v);
    // }

    Ok(())
}
