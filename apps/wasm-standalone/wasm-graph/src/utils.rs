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

use super::types::*;
use serde_json;
use std::ptr;

pub unsafe fn load_input(in_addr: i32, in_size: usize) -> Tensor {
    let in_addr = in_addr as *mut u8;

    println!("DEBUG: in_addr {:?}, in_size {:?}", in_addr, in_size);

    let data_vec = unsafe { std::slice::from_raw_parts(in_addr, in_size) };

    let input = serde_json::from_slice(&data_vec);
    match input {
        Ok(result) => {
            println!("DEBUG: SER SUCCEED!!! and Ok");
            result
        }
        Err(e) => {
            panic!("DEBUG: SER SUCCEED!!! but Err, {:?}", &e);
        }
    }
}

pub unsafe fn store_output(out_addr: i32, output: Tensor) -> usize {
    let out_addr = out_addr as *mut u8;

    let data_vec = serde_json::to_vec(&output).unwrap();
    let data_size = data_vec.len();
    for i in 0..data_size {
        ptr::write(out_addr.offset(i as isize), *data_vec.get(i).unwrap());
    }

    data_size
}
