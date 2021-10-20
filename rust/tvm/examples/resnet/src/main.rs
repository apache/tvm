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
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

use ::ndarray::{Array, ArrayD, Axis};
use image::{FilterType, GenericImageView};

use anyhow::Context as _;
use tvm_rt::graph_rt::GraphRt;
use tvm_rt::*;

fn main() -> anyhow::Result<()> {
    let dev = Device::cpu(0);
    println!("{}", concat!(env!("CARGO_MANIFEST_DIR"), "/cat.png"));

    let img = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/cat.png"))
        .context("Failed to open cat.png")?;

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
    let input = NDArray::from_rust_ndarray(&arr, Device::cpu(0), DataType::float(32, 1))?;
    println!(
        "input shape is {:?}, len: {}, size: {}",
        input.shape(),
        input.len(),
        input.size(),
    );

    let graph = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_graph.json"))
        .context("Failed to open graph")?;

    // load the built module
    let lib = Module::load(&Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/deploy_lib.so"
    )))?;

    // parse parameters and convert to TVMByteArray
    let params: Vec<u8> = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_param.params"))?;
    println!("param bytes: {}", params.len());

    // If you want an easy way to test a memory leak simply replace the program below with:
    // let mut output: Vec<f32>;

    // loop {
    //     let mut graph_rt = GraphRt::create_from_parts(&graph, lib.clone(), dev)?;
    //     graph_rt.load_params(params.clone())?;
    //     graph_rt.set_input("data", input.clone())?;
    //     graph_rt.run()?;

    //     // prepare to get the output
    //     let output_shape = &[1, 1000];
    //     let output_nd = NDArray::empty(output_shape, Device::cpu(0), DataType::float(32, 1));
    //     graph_rt.get_output_into(0, output_nd.clone())?;

    //     // flatten the output as Vec<f32>
    //     output = output_nd.to_vec::<f32>()?;
    // }

    let mut graph_rt = GraphRt::create_from_parts(&graph, lib, dev)?;
    graph_rt.load_params(params)?;
    graph_rt.set_input("data", input)?;
    graph_rt.run()?;

    // prepare to get the output
    let output_shape = &[1, 1000];
    let output_nd = NDArray::empty(output_shape, Device::cpu(0), DataType::float(32, 1));
    graph_rt.get_output_into(0, output_nd.clone())?;

    // flatten the output as Vec<f32>
    let output: Vec<f32> = output_nd.to_vec::<f32>()?;

    // find the maximum entry in the output and its index
    let (argmax, max_prob) = output
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // create a hash map of (class id, class name)
    let file = File::open("synset.txt").context("failed to open synset")?;
    let synset: Vec<std::string::String> = BufReader::new(file)
        .lines()
        .into_iter()
        .map(|x| x.expect("readline failed"))
        .collect();

    let label = &synset[argmax];
    println!(
        "input image belongs to the class `{}` with probability {}",
        label, max_prob
    );

    Ok(())
}
