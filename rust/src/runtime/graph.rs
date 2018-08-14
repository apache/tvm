use std::{cmp, collections::HashMap, convert::TryFrom, iter::FromIterator, mem, str};

use nom::{alpha1, digit1, le_i32, le_i64, le_u16, le_u32, le_u64, le_u8, types::CompleteStr};
use serde;
use serde_json;

use super::{DataType, Module, Storage, TVMArgValue, TVMContext, Tensor};
use errors::{Error, ErrorKind, Result};
use ffi::runtime::{
  DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt, DLTensor,
};

const NDARRAY_MAGIC: u64 = 0xDD5E40F096B4A13F; // Magic number for NDArray file
const NDARRAY_LIST_MAGIC: u64 = 0xF7E58D4F05049CB7; // Magic number for NDArray list file

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
  fn entry_index(&self, entry: &Entry) -> Result<usize> {
    self
      .node_row_ptr
      .as_ref()
      .map(|nrp| nrp[entry.id] + entry.index)
      .ok_or("Missing node_row_ptr.".into())
  }

  fn get_attr<T: serde::de::DeserializeOwned>(&self, attr: &str) -> Result<T> {
    Ok(serde_json::from_value::<T>(
      self
        .attrs
        .as_ref()
        .ok_or(ErrorKind::GraphFormatError(
          "Missing graph attrs".to_string(),
        ))?
        .get(attr)
        .ok_or(ErrorKind::GraphFormatError(format!(
          "Missing {} attr",
          attr
        )))?
        .to_owned(),
    )?)
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

impl Node {
  fn parse_attrs(&self) -> Result<NodeAttrs> {
    let attrs = self
      .attrs
      .as_ref()
      .ok_or(format!("Missing node.attrs for `{}`", self.name))?;
    let func_name = attrs
      .get("func_name")
      .ok_or(format!("Node `{}` is missing attrs.func_name", self.name))?
      .to_string();
    let num_outputs = attrs
      .get("num_outputs")
      .ok_or(format!("Node `{}` is missing attrs.num_outputs", self.name))?
      .parse::<usize>()?;
    let flatten_data = attrs
      .get("flatten_data")
      .ok_or(format!(
        "Node `{}` is missing attrs.flatten_data",
        self.name
      ))?
      .parse::<u8>()? == 1;
    Ok(NodeAttrs {
      func_name,
      num_outputs,
      flatten_data,
    })
  }
}

impl<'a> TryFrom<&'a String> for Graph {
  type Error = Error;
  fn try_from(graph_json: &String) -> Result<Self> {
    let graph = serde_json::from_str(graph_json)?;
    Ok(graph)
  }
}

impl<'a> TryFrom<&'a str> for Graph {
  type Error = Error;
  fn try_from(graph_json: &'a str) -> Result<Self> {
    let graph = serde_json::from_str(graph_json)?;
    Ok(graph)
  }
}

pub struct GraphExecutor<'m, 't> {
  graph: Graph,
  op_execs: Vec<Box<Fn() + 'm>>,
  tensors: Vec<Tensor<'t>>,
}

impl<'m, 't> GraphExecutor<'m, 't> {
  pub fn new<M: 'm + Module>(graph: Graph, lib: &'m M) -> Result<Self> {
    let tensors = Self::setup_storages(&graph)?;
    Ok(GraphExecutor {
      op_execs: Self::setup_op_execs(&graph, lib, &tensors)?,
      tensors: tensors,
      graph: graph,
    })
  }

  pub fn run(&self) {
    self.op_execs.iter().for_each(|op_exec| {
      op_exec();
    });
  }

  fn setup_storages<'a>(graph: &'a Graph) -> Result<Vec<Tensor<'t>>> {
    let storage_ids = graph.get_attr::<(String, Vec<usize>)>("storage_id")?.1;
    let shapes = graph.get_attr::<(String, Vec<Vec<usize>>)>("shape")?.1;
    let dtypes = graph
      .get_attr::<(String, Vec<String>)>("dltype")?
      .1
      .iter()
      .map(|dltype| {
        if let Ok((_, dtype)) = tvm_str_to_type(CompleteStr(dltype)) {
          Ok(dtype)
        } else {
          Err(ErrorKind::GraphFormatError(format!("Invalid dltype: {}", dltype).to_string()).into())
        }
      })
      .collect::<Result<Vec<DataType>>>()?;

    let align = dtypes.iter().map(|dtype| dtype.bits as usize >> 3).max();
    let mut storage_num_bytes = vec![0usize; *storage_ids.iter().max().unwrap_or(&1) + 1];
    for (i, &storage_id) in storage_ids.iter().enumerate() {
      let dtype_size = dtypes[i].bits * dtypes[i].lanes >> 3;
      let nbytes = dtype_size * shapes[i].iter().product::<usize>();
      storage_num_bytes[storage_id] = cmp::max(nbytes, storage_num_bytes[storage_id]);
    }

    let mut storages: Vec<Storage> = storage_num_bytes
      .into_iter()
      .map(|nbytes| Storage::new(nbytes, align))
      .collect::<Result<Vec<Storage>>>()?;

    let tensors = izip!(storage_ids, shapes, dtypes)
      .map(|(storage_id, shape, dtype)| {
        let storage = storages[storage_id].view();
        Tensor {
          data: mem::replace(&mut storages[storage_id], storage),
          ctx: TVMContext::default(),
          dtype: dtype,
          numel: shape.iter().product(),
          dshape: shape.iter().map(|&v| v as i64).collect(),
          shape: shape,
          strides: None,
          byte_offset: 0,
        }
      })
      .collect();

    Ok(tensors)
  }

  fn setup_op_execs<M: 'm + Module>(
    graph: &Graph,
    lib: &'m M,
    tensors: &Vec<Tensor<'t>>,
  ) -> Result<Vec<Box<Fn() + 'm>>> {
    ensure!(graph.node_row_ptr.is_some(), "Missing node_row_ptr.");
    let node_row_ptr = graph.node_row_ptr.as_ref().unwrap();
    graph
      .nodes
      .iter()
      .enumerate()
      .filter(|(_i, node)| node.op != "null")
      .map(|(i, node)| {
        ensure!(node.op == "tvm_op", "Only TVM ops are supported.");
        ensure!(node.attrs.is_some(), "Missing node_row_ptr.");
        let attrs = node.parse_attrs()?;
        let func = lib
          .get_function(&attrs.func_name)
          .ok_or(format!("Missing function {}", attrs.func_name))?;
        let arg_indices = node
          .inputs
          .iter()
          .map(|entry| graph.entry_index(entry))
          .chain((0..attrs.num_outputs).map(|oi| Ok(node_row_ptr[i].clone() + oi)));

        let dl_tensors = arg_indices
          .map(|idx| {
            let tensor = &tensors[idx?];
            Ok(if attrs.flatten_data {
              DLTensor::from_tensor(tensor, true /* flatten */)
            } else {
              DLTensor::from(tensor)
            })
          })
          .collect::<Result<Vec<DLTensor>>>()
          .unwrap();
        let op: Box<Fn()> = box move || {
          let args = dl_tensors
            .iter()
            .map(|t| t.into())
            .collect::<Vec<TVMArgValue>>();
          func(args.as_slice());
        };
        Ok(op)
      })
      .collect()
  }

  pub fn load_params(&mut self, params: HashMap<String, Tensor<'t>>) {
    params.into_iter().for_each(|(name, param)| {
      self.set_input(name, param);
    })
  }

  pub fn set_input<S: AsRef<str>>(&mut self, name: S, value: Tensor<'t>) {
    if let Some(idx) = self.get_input_index(name.as_ref()) {
      // TODO: consider `new_with_params` to avoid ever allocating
      let ptr = self.tensors[idx].data.as_ptr();
      let mut to_replace = self.tensors.iter_mut().filter(|t| t.data.as_ptr() == ptr);
      let mut owner = to_replace.nth(0).unwrap();
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

  pub fn get_input<S: AsRef<str>>(&mut self, name: S) -> Option<&Tensor> {
    self
      .get_input_index(name.as_ref())
      .and_then(move |idx| Some(&self.tensors[idx]))
  }

  pub fn get_output(&self, idx: usize) -> Option<&Tensor> {
    let graph = &self.graph;
    graph.heads.get(idx).and_then(|entry| {
      graph
        .entry_index(entry)
        .map(|idx| self.tensors.get(idx))
        .unwrap_or(None)
    })
  }

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

named!(
  tvm_str_to_type<CompleteStr, DataType>,
  do_parse!(
    type_name: alpha1 >>
    bits: digit1 >>
    lanes: opt!(tuple!(tag!("x"), digit1)) >>
    (DataType {
      code: match type_name {
        CompleteStr("int") => DLDataTypeCode_kDLInt,
        CompleteStr("uint") => DLDataTypeCode_kDLUInt,
        CompleteStr("float") => DLDataTypeCode_kDLFloat,
        _ => DLDataTypeCode_kDLFloat,
      } as usize,
      bits: bits.parse::<u8>().unwrap() as usize,
      lanes: match lanes {
        Some(lanes) => lanes.1.parse::<u16>().unwrap() as usize,
        None => 1,
      },
    })
  )
);

named!(
  name<String>,
  map_res!(length_bytes!(le_u64), |b: &[u8]| {
    String::from_utf8(b.to_vec())
  })
);

named!(
  tvm_ctx<&[u8], TVMContext>,
  do_parse!(
    device_type: le_u32 >>
    device_id: le_i32 >>
    (TVMContext { device_type: device_type as usize, device_id: device_id as usize })
  )
);

named!(
  data_type<&[u8], DataType>,
  do_parse!(
    code: le_u8 >>
    bits: le_u8 >>
    lanes: le_u16 >>
    (DataType { code: code as usize, bits: bits as usize, lanes: lanes as usize })
  )
);

named!(
  tensor<Tensor>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> ctx: tvm_ctx >> ndim: le_u32 >> dtype: data_type
      >> shape: count!(map!(le_i64, |sz| sz as usize), ndim as usize) >> length: le_i64
      >> data: take!(length) >> (Tensor {
      data: Storage::from(data),
      ctx: ctx,
      dtype: dtype,
      numel: shape.iter().product(),
      dshape: shape.iter().map(|&v| v as i64).collect(),
      shape: shape,
      strides: None,
      byte_offset: 0,
    })
  )
);

named!(
  parse_param_dict<HashMap<String, Tensor>>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> names: length_count!(le_u64, name)
      >> tensors: length_count!(le_u64, tensor)
      >> (HashMap::from_iter(names.into_iter().zip(tensors.into_iter())))
  )
);

pub fn load_param_dict(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
  if let Ok((remaining_bytes, param_dict)) = parse_param_dict(bytes) {
    if remaining_bytes.len() > 0 {
      bail!(ErrorKind::LoadGraphParamsError("extra input".to_string()))
    } else {
      Ok(param_dict)
    }
  } else {
    bail!(ErrorKind::LoadGraphParamsError(
      "invalid parameters file".to_string()
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_str_to_type() {
    assert_eq!(
      tvm_str_to_type(CompleteStr("float24")).unwrap().1,
      DataType {
        code: DLDataTypeCode_kDLFloat as usize,
        bits: 24,
        lanes: 1
      }
    );
    assert_eq!(
      tvm_str_to_type(CompleteStr("uint111x44")).unwrap().1,
      DataType {
        code: DLDataTypeCode_kDLUInt as usize,
        bits: 111,
        lanes: 44
      }
    );
  }
}
