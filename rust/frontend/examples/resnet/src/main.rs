#![feature(try_from)]

extern crate csv;
extern crate image;
extern crate ndarray;
extern crate tvm_frontend as tvm;

use std::{
    collections::HashMap,
    convert::TryInto,
    error::Error,
    fs::{self, File},
    path::Path,
    result::Result,
};

use image::{FilterType, GenericImageView};
use ndarray::{Array, ArrayD, Axis};

use tvm::*;

fn main() -> Result<(), Box<Error>> {
    let ctx = TVMContext::cpu(0);
    let img = image::open("cat.png")?;
    println!("original image dimensions: {:?}", img.dimensions());
    // for bigger size images, one needs to first resize to 256x256
    // with `img.resize_exact` method and then `image.crop` to 224x224
    let img = img.resize(224, 224, FilterType::Nearest).to_rgb();
    println!("resized image dimensions: {:?}", img.dimensions());
    let mut pixels: Vec<f32> = vec![];
    for pixel in img.pixels() {
        let tmp = pixel.data;
        // normalize the RGB channels using mean, std of imagenet1k
        let tmp = [
            (tmp[0] as f32 - 123.0) / 58.395, // R
            (tmp[1] as f32 - 117.0) / 57.12,  // G
            (tmp[2] as f32 - 104.0) / 57.375, // B
        ];
        for e in &tmp {
            pixels.push(*e);
        }
    }

    let arr = Array::from_shape_vec((224, 224, 3), pixels)?;
    let arr: ArrayD<f32> = arr.permuted_axes([2, 0, 1]).into_dyn();
    // make arr shape as [1, 3, 224, 224] acceptable to resnet
    let arr = arr.insert_axis(Axis(0));
    // create input tensor from rust's ndarray
    let input = NDArray::from_rust_ndarray(&arr, TVMContext::cpu(0), TVMType::from("float32"))?;
    println!(
        "input size is {:?}",
        input.shape().expect("cannot get the input shape")
    );
    let graph = fs::read_to_string("deploy_graph.json")?;
    // load the built module
    let lib = Module::load(&Path::new("deploy_lib.so"))?;
    // get the global TVM graph runtime function
    let runtime_create_fn = Function::get_function("tvm.graph_runtime.create", true).unwrap();
    let runtime_create_fn_ret = call_packed!(
        runtime_create_fn,
        &graph,
        &lib,
        &ctx.device_type,
        &ctx.device_id
    )?;
    // get graph runtime module
    let graph_runtime_module: Module = runtime_create_fn_ret.try_into()?;
    // get the registered `load_params` from runtime module
    let load_param_fn = graph_runtime_module
        .get_function("load_params", false)
        .unwrap();
    // parse parameters and convert to TVMByteArray
    let params: Vec<u8> = fs::read("deploy_param.params")?;
    let barr = TVMByteArray::from(&params);
    // load the parameters
    call_packed!(load_param_fn, &barr)?;
    // get the set_input function
    let set_input_fn = graph_runtime_module
        .get_function("set_input", false)
        .unwrap();

    call_packed!(set_input_fn, "data", &input)?;
    // get `run` function from runtime module
    let run_fn = graph_runtime_module.get_function("run", false).unwrap();
    // execute the run function. Note that it has no argument
    call_packed!(run_fn,)?;
    // prepare to get the output
    let output_shape = &mut [1, 1000];
    let output = empty(output_shape, TVMContext::cpu(0), TVMType::from("float32"));
    // get the `get_output` function from runtime module
    let get_output_fn = graph_runtime_module
        .get_function("get_output", false)
        .unwrap();
    // execute the get output function
    call_packed!(get_output_fn, &0, &output)?;
    // flatten the output as Vec<f32>
    let output = output.to_vec::<f32>()?;
    // find the maximum entry in the output and its index
    let mut argmax = -1;
    let mut max_prob = 0.;
    for i in 0..output.len() {
        if output[i] > max_prob {
            max_prob = output[i];
            argmax = i as i32;
        }
    }
    // create a hash map of (class id, class name)
    let mut synset: HashMap<i32, String> = HashMap::new();
    let file = File::open("synset.csv")?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    for result in rdr.records() {
        let record = result?;
        let id: i32 = record[0].parse()?;
        let cls = record[1].to_string();
        synset.insert(id, cls);
    }

    println!(
        "input image belongs to the class `{}` with probability {}",
        synset
            .get(&argmax)
            .expect("cannot find the class id for argmax"),
        max_prob
    );

    Ok(())
}
