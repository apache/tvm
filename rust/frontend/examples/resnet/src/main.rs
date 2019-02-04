#![feature(try_from)]

extern crate csv;
extern crate image;
extern crate ndarray;
extern crate tvm_frontend as tvm;

use std::{
    collections::HashMap,
    convert::TryInto,
    fs::{self, File},
    path::Path,
};

use image::{FilterType, GenericImageView};
use ndarray::{Array, ArrayD, Axis};

use tvm::*;
use tvm::graph_runtime::GraphModule;

fn main() {
    let ctx = TVMContext::cpu(0);
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
    let input =
        NDArray::from_rust_ndarray(&arr, TVMContext::cpu(0), TVMType::from("float32")).unwrap();
    println!(
        "input size is {:?}",
        input.shape().expect("cannot get the input shape")
    );
    let mut module = GraphModule::from_paths(
        concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_graph.json"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_lib.so"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_param.params"),
        ctx
    ).unwrap();
    let mut result: Vec<NDArray> = module.apply(&[&input]);
    let output = result.pop().unwrap();

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
