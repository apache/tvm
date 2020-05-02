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

use std::{cmp, collections::HashMap, convert::TryFrom, iter::FromIterator, mem, str};

use failure::Error;
use nom::{
    character::complete::{alpha1, digit1},
    number::complete::{le_i32, le_i64, le_u16, le_u32, le_u64, le_u8},
};
use serde;
use serde_json;
use tvm_common::{
    array::{DataType, TVMContext},
    ffi::{DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt, DLTensor},
    TVMArgValue,
};

use crate::{errors::GraphFormatError, Module, Storage, Tensor};

// @see `kTVMNDArrayMagic` in `ndarray.h`
const _NDARRAY_MAGIC: u64 = 0xDD5E_40F0_96B4_A13F;
// @see `kTVMNDArrayListMagic` in `graph_runtime.h`
const _NDARRAY_LIST_MAGIC: u64 = 0xF7E5_8D4F_0504_9CB7;

/// A TVM computation graph.
///
/// # Examples
///
/// ```norun
/// let graph_json = fs::read_to_string("graph.json").unwrap();
/// let graph = Graph::try_from(&graph_json).unwrap();
/// ```
#[derive(Serialize, Deserialize, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub arg_nodes: Vec<usize>,
    pub heads: Vec<Entry>,
    pub node_row_ptr: Option<Vec<usize>>,
    pub attrs: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Entry {
    pub id: usize,
    pub index: usize,
    pub version: usize,
}

impl Graph {
    fn entry_index(&self, entry: &Entry) -> Result<usize, GraphFormatError> {
        self.node_row_ptr
            .as_ref()
            .map(|nrp| nrp[entry.id] + entry.index)
            .ok_or_else(|| GraphFormatError::MissingField("node_row_ptr"))
    }

    /// Attempt to deserialize a JSON attribute to a type `T`.
    fn get_attr<T: serde::de::DeserializeOwned>(&self, attr: &str) -> Result<T, GraphFormatError> {
        Ok(serde_json::from_value::<T>(
            self.attrs
                .as_ref()
                .ok_or(GraphFormatError::MissingField("attrs"))?
                .get(attr)
                .ok_or_else(|| {
                    GraphFormatError::MissingAttr("graph".to_string(), attr.to_string())
                })?
                .to_owned(),
        )
        .map_err(|err| GraphFormatError::Parse(err.into()))?)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Node {
    pub op: String,
    pub name: String,
    pub inputs: Vec<Entry>,
    pub attrs: Option<HashMap<String, String>>,
    pub control_deps: Option<Vec<Entry>>,
}

struct NodeAttrs {
    func_name: String,
    num_outputs: usize,
    flatten_data: bool,
}

macro_rules! get_node_attr {
    ($node:expr, $attrs:ident, $attr:literal) => {
        $attrs
            .get($attr)
            .ok_or_else(|| GraphFormatError::MissingAttr($node.to_owned(), $attr.to_owned()))
    };
}

impl Node {
    fn parse_attrs(&self) -> Result<NodeAttrs, Error> {
        let attrs = self
            .attrs
            .as_ref()
            .ok_or_else(|| GraphFormatError::MissingAttr(self.name.clone(), "attrs".to_owned()))?;
        Ok(NodeAttrs {
            func_name: get_node_attr!(self.name, attrs, "func_name")?.to_owned(),
            num_outputs: get_node_attr!(self.name, attrs, "num_outputs")?.parse::<usize>()?,
            flatten_data: get_node_attr!(self.name, attrs, "flatten_data")?.parse::<u8>()? == 1,
        })
    }
}

impl<'a> TryFrom<&'a String> for Graph {
    type Error = Error;
    fn try_from(graph_json: &String) -> Result<Self, self::Error> {
        let graph = serde_json::from_str(graph_json)?;
        Ok(graph)
    }
}

impl<'a> TryFrom<&'a str> for Graph {
    type Error = Error;
    fn try_from(graph_json: &'a str) -> Result<Self, Self::Error> {
        let graph = serde_json::from_str(graph_json)?;
        Ok(graph)
    }
}

/// A executor for a TVM computation graph.
///
/// # Examples
///
/// ```norun
/// use ndarray::Array;
///
/// let syslib = SystemLibModule::default(); // a provider of TVM functions
///
/// let mut params_bytes = Vec::new();
/// fs::File::open("graph.params").unwrap().read_to_end(&mut params_bytes).unwrap();
/// let params = tvm::runtime::load_param_dict(&params_bytes).unwrap();
///
/// let graph = Graph::try_from(&fs::read_to_string("graph.json").unwrap()).unwrap();
///
/// let mut exec = GraphExecutor::new(graph, &syslib).unwrap();
/// exec.load_params(params);
///
/// let x = Array::from_vec(vec![1f32, 2., 3., 4.]);
/// exec.set_input("data", x.into());
/// exec.run();
/// let output = exec.get_output(0).unwrap();
///
/// println!("{:#?}", Array::try_from(output).unwrap());
/// ```
pub struct GraphExecutor<'m, 't> {
    graph: Graph,
    op_execs: Vec<Box<dyn Fn() + 'm>>,
    tensors: Vec<Tensor<'t>>,
}

unsafe impl<'m, 't> Send for GraphExecutor<'m, 't> {}

impl<'m, 't> GraphExecutor<'m, 't> {
    pub fn new<M: 'm + Module>(graph: Graph, lib: &'m M) -> Result<Self, Error> {
        let tensors = Self::setup_storages(&graph)?;
        Ok(GraphExecutor {
            op_execs: Self::setup_op_execs(&graph, lib, &tensors)?,
            tensors,
            graph,
        })
    }

    /// Runs the computation graph.
    pub fn run(&mut self) {
        self.op_execs.iter().for_each(|op_exec| {
            op_exec();
        });
    }

    /// Allocates `Storages` for each `storage_id` and returns `Tensor`s to hold each output.
    fn setup_storages<'a>(graph: &'a Graph) -> Result<Vec<Tensor<'t>>, Error> {
        let storage_ids = graph.get_attr::<(String, Vec<usize>)>("storage_id")?.1;
        let shapes = graph.get_attr::<(String, Vec<Vec<i64>>)>("shape")?.1;
        let dtypes = graph
            .get_attr::<(String, Vec<String>)>("dltype")?
            .1
            .iter()
            .map(|dltype| {
                if let Ok((_, dtype)) = tvm_str_to_type(dltype) {
                    Ok(dtype)
                } else {
                    Err(GraphFormatError::InvalidDLType(dltype.to_string()))
                }
            })
            .collect::<Result<Vec<DataType>, GraphFormatError>>()?;

        let align = dtypes.iter().map(|dtype| dtype.bits() as usize).max();
        let mut storage_num_bytes = vec![0usize; *storage_ids.iter().max().unwrap_or(&1) + 1];
        for (i, &storage_id) in storage_ids.iter().enumerate() {
            let dtype_size = (dtypes[i].bits() * dtypes[i].lanes()) >> 3;
            let nbytes = dtype_size * shapes[i].iter().product::<i64>() as usize;
            storage_num_bytes[storage_id] = cmp::max(nbytes, storage_num_bytes[storage_id]);
        }

        let mut storages: Vec<Storage> = storage_num_bytes
            .into_iter()
            .map(|nbytes| Storage::new(nbytes, align))
            .collect::<Result<Vec<Storage>, Error>>()?;

        let tensors = izip!(storage_ids, shapes, dtypes)
            .map(|(storage_id, shape, dtype)| {
                let storage = storages[storage_id].view();
                Tensor {
                    data: mem::replace(&mut storages[storage_id], storage),
                    ctx: TVMContext::default(),
                    dtype,
                    size: shape.iter().product::<i64>() as usize,
                    shape,
                    strides: None,
                    byte_offset: 0,
                }
            })
            .collect();

        Ok(tensors)
    }

    /// Creates closures which represent the computation performed by this graph.
    fn setup_op_execs<M: 'm + Module>(
        graph: &Graph,
        lib: &'m M,
        tensors: &[Tensor<'t>],
    ) -> Result<Vec<Box<dyn Fn() + 'm>>, Error> {
        ensure!(graph.node_row_ptr.is_some(), "Missing node_row_ptr.");
        let node_row_ptr = graph.node_row_ptr.as_ref().unwrap();

        let mut op_execs = Vec::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            if node.op == "null" {
                continue;
            }
            ensure!(node.op == "tvm_op", "Only TVM ops are supported.");
            ensure!(node.attrs.is_some(), "Missing node attrs.");

            let attrs = node.parse_attrs()?;

            if attrs.func_name == "__nop" {
                continue;
            }

            let func = lib
                .get_function(&attrs.func_name)
                .ok_or_else(|| format_err!("Library is missing function {}", attrs.func_name))?;
            let arg_indices = node
                .inputs
                .iter()
                .map(|entry| graph.entry_index(entry))
                .chain((0..attrs.num_outputs).map(|oi| Ok(node_row_ptr[i] + oi)));

            let dl_tensors = arg_indices
                .map(|idx| {
                    let tensor = &tensors[idx?];
                    Ok(if attrs.flatten_data {
                        Tensor::as_dltensor(tensor, true /* flatten */)
                    } else {
                        DLTensor::from(tensor)
                    })
                })
                .collect::<Result<Vec<DLTensor>, Error>>()
                .unwrap();
            let op: Box<dyn Fn()> = Box::new(move || {
                let args = dl_tensors
                    .iter()
                    .map(|t| t.into())
                    .collect::<Vec<TVMArgValue>>();
                func(&args).unwrap();
            });
            op_execs.push(op);
        }
        Ok(op_execs)
    }

    pub fn load_params(&mut self, params: HashMap<String, Tensor>) {
        params.into_iter().for_each(|(name, param)| {
            self.set_input(name, param);
        })
    }

    #[allow(clippy::if_same_then_else)]
    pub fn set_input<S: AsRef<str>>(&mut self, name: S, value: Tensor) {
        if let Some(idx) = self.get_input_index(name.as_ref()) {
            // TODO: consider `new_with_params` to avoid ever allocating
            let ptr = self.tensors[idx].data.as_ptr();
            let mut to_replace = self.tensors.iter_mut().filter(|t| t.data.as_ptr() == ptr);
            let owner = to_replace.nth(0).unwrap();
            if value.data.is_owned() {
                // FIXME: for no-copy, need setup_op_execs to not capture tensor ptr
                // mem::replace(&mut (*owner), value);
                // to_replace.for_each(|t| {
                //   panic!("replacing");
                //   t.data = owner.data.view();
                // });
                owner.copy(&value);
            } else {
                owner.copy(&value);
            }
        } else {
            println!("Unexpected input `{}`", name.as_ref());
        }
    }

    /// Returns the graph input with name `name`, if it exists.
    pub fn get_input<S: AsRef<str>>(&mut self, name: S) -> Option<&Tensor> {
        self.get_input_index(name.as_ref())
            .map(move |idx| &self.tensors[idx])
    }

    /// Returns the graph output with index `index`, if it exists.
    pub fn get_output(&self, idx: usize) -> Option<&Tensor> {
        let graph = &self.graph;
        graph.heads.get(idx).and_then(|entry| {
            graph
                .entry_index(entry)
                .map(|idx| self.tensors.get(idx))
                .unwrap_or(None)
        })
    }

    /// Returns the index for graph input with name `name`, if it exists.
    pub fn get_input_index<S: AsRef<str>>(&self, name: S) -> Option<usize> {
        let graph = &self.graph;
        (0..graph.nodes.len())
            .skip_while(|&i| graph.nodes[i].name != name.as_ref())
            .nth(0)
            .and_then(|i| {
                if graph.arg_nodes.iter().any(|&id| id == i) {
                    graph.node_row_ptr.as_ref().map(|nrp| nrp[i])
                } else {
                    None
                }
            })
    }
}

// Converts a string to TVM DLDataTypeCode. @see `String2DLDataType` in packed_func.h
named! {
  tvm_str_to_type<&str, DataType>,
  do_parse!(
    type_name: alpha1 >>
    bits:      digit1 >>
    lanes:     opt!(complete!(tuple!(tag!("x"), digit1))) >>
    (
        DataType {
            code: match type_name {
                "int" => DLDataTypeCode_kDLInt,
                "uint" => DLDataTypeCode_kDLUInt,
                "float" => DLDataTypeCode_kDLFloat,
                _ => DLDataTypeCode_kDLFloat,
            } as usize,
            bits: bits.parse::<u8>().unwrap() as usize,
            lanes: lanes
                .map(|(_, lanes)| lanes.parse::<u16>().unwrap() as usize)
                .unwrap_or(1)
        }
    )
  )
}

// Converts a bytes to String.
named! {
    name<String>,
    do_parse!(
        len_l: le_u32 >>
        len_h: le_u32 >>
        data: take!(len_l) >>
        (
            if len_h == 0 {
                String::from_utf8(data.to_vec()).unwrap()
            } else {
                panic!("Too long string")
            }
        )
    )
}

// Parses a TVMContext
named! {
  tvm_ctx<&[u8], TVMContext>,
  do_parse!(
    device_type: le_u32 >>
    device_id:   le_i32 >>
    (
        TVMContext {
            device_type: device_type as usize,
            device_id: device_id as usize,
        }
    )
  )
}

// Parses a DataType
named! {
  data_type<&[u8], DataType>,
  do_parse!(
    code:  le_u8  >>
    bits:  le_u8  >>
    lanes: le_u16 >>
    (DataType { code: code as usize, bits: bits as usize, lanes: lanes as usize })
  )
}

// Parses a Tensor from a TVM array file.
named! {
    tensor<Tensor>,
    do_parse!(
                take!(8)      >>
                le_u64        >>
        ctx:    tvm_ctx       >>
        ndim:   le_u32        >>
        dtype:  data_type     >>
        shape:  count!(map!(le_i64, |sz| sz as i64), ndim as usize) >>
        length: le_i64        >>
        data:   take!(length) >>
        (
            Tensor {
                data: Storage::from(data),
                ctx: ctx,
                dtype: dtype,
                size: shape.iter().product::<i64>() as usize,
                shape: shape,
                strides: None,
                byte_offset: 0,
            }
        )
    )
}

// Parses a graph params dict from a params binary file.
named! {
    parse_param_dict<HashMap<String, Tensor>>,
    do_parse!(
                 take!(8)                      >>
                 le_u64                        >>
        names:   length_count!(le_u64, name)   >>
        tensors: length_count!(le_u64, tensor) >>
        (
            HashMap::from_iter(names.into_iter().zip(tensors.into_iter()))
        )
    )
}

/// Loads a param dict saved using `relay.save_param_dict`.
pub fn load_param_dict(bytes: &[u8]) -> Result<HashMap<String, Tensor>, GraphFormatError> {
    if let Ok((remaining_bytes, param_dict)) = parse_param_dict(bytes) {
        if remaining_bytes.is_empty() {
            Ok(param_dict)
        } else {
            Err(GraphFormatError::Params)
        }
    } else {
        Err(GraphFormatError::Params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_to_type() {
        assert_eq!(
            tvm_str_to_type("float24").unwrap().1,
            DataType {
                code: DLDataTypeCode_kDLFloat as usize,
                bits: 24,
                lanes: 1
            }
        );
        assert_eq!(
            tvm_str_to_type("uint111x44").unwrap().1,
            DataType {
                code: DLDataTypeCode_kDLUInt as usize,
                bits: 111,
                lanes: 44
            }
        );
    }
}
