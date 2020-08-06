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

use std::{
    collections::HashMap,
    convert::TryInto,
    fs::{self, File},
    path::Path,
};

use ::ndarray::{Array, ArrayD, Axis};
use image::{FilterType, GenericImageView};

use tvm::runtime::ByteArray;
use tvm::*;

fn main() {
    let ctx = Context::cpu(0);
    let img = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/cat.png")).unwrap();
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

    let arr = Array::from_shape_vec((224, 224, 3), pixels).unwrap();
    let arr: ArrayD<f32> = arr.permuted_axes([2, 0, 1]).into_dyn();
    // make arr shape as [1, 3, 224, 224] acceptable to resnet
    let arr = arr.insert_axis(Axis(0));
    // create input tensor from rust's ndarray
    let input = NDArray::from_rust_ndarray(&arr, Context::cpu(0), DataType::float(32, 1)).unwrap();
    println!(
        "input size is {:?}",
        input.shape().expect("cannot get the input shape")
    );
    let graph =
        fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_graph.json")).unwrap();
    // load the built module
    let lib = Module::load(&Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/deploy_lib.so"
    )))
    .unwrap();
    // get the global TVM graph runtime function
    let runtime_create_fn = Function::get("tvm.graph_runtime.create").unwrap();
    let runtime_create_fn_ret = runtime_create_fn.invoke(vec![
        graph.into(),
        (&lib).into(),
        (&ctx.device_type).into(),
        (&ctx.device_id).into(),
    ]);

    // get graph runtime module
    let graph_runtime_module: Module = runtime_create_fn_ret.unwrap().try_into().unwrap();

    // get the registered `load_params` from runtime module
    let ref load_param_fn = graph_runtime_module
        .get_function("load_params", false)
        .unwrap();
    // parse parameters and convert to TVMByteArray
    let params: Vec<u8> =
        fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_param.params")).unwrap();
    let barr = ByteArray::from(&params);
    // load the parameters
    load_param_fn.invoke(vec![(&barr).into()]).unwrap();
    // get the set_input function
    let ref set_input_fn = graph_runtime_module
        .get_function("set_input", false)
        .unwrap();

    set_input_fn
        .invoke(vec!["data".into(), (&input).into()])
        .unwrap();

    // get `run` function from runtime module
    let ref run_fn = graph_runtime_module.get_function("run", false).unwrap();
    // execute the run function. Note that it has no argument
    run_fn.invoke(vec![]).unwrap();
    // prepare to get the output
    let output_shape = &mut [1, 1000];
    let output = NDArray::empty(output_shape, Context::cpu(0), DataType::float(32, 1));
    // get the `get_output` function from runtime module
    let ref get_output_fn = graph_runtime_module
        .get_function("get_output", false)
        .unwrap();
    // execute the get output function
    get_output_fn
        .invoke(vec![(&0).into(), (&output).into()])
        .unwrap();
    // flatten the output as Vec<f32>
    let output = output.to_vec::<f32>().unwrap();
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
    let file = File::open("synset.csv").unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    for result in rdr.records() {
        let record = result.unwrap();
        let id: i32 = record[0].parse().unwrap();
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
}
