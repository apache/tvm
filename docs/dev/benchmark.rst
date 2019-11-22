..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

********************************
Benchmark Performance Log Format
********************************
This page details schema v0.1 for a unified benchmark log format. This schema will allow easier cross-references with other frameworks/runs, experiment reproduction, data for nightly perf regression, and the separation of logging/visualization efforts.

Log Format Overview
~~~~~~~~~~~~~~~~~~~

+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| header                | examples                                                                                                                                                   | category     | notes/justification                                                          |
+=======================+============================================================================================================================================================+==============+==============================================================================+
| timestamp             | 1572282699.6                                                                                                                                               | metadata     | indicates when this record is logged                                         |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| schema\_version       | 0.1                                                                                                                                                        | metadata     | ensure reproducibility as we iterate on this schema                          |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| metadata              | { "docker\_tag":"gcr.io/.../0a680", ... }                                                                                                                  | metadata     | ``docker_tag`` is required                                                   |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| workload              | resnet-18                                                                                                                                                  | workload     | name of workload                                                             |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| workload\_args        | {“input\_name”: "Input3", “input\_shape”: [list\_of\_shape], “data\_layout”: NHCW}                                                                         | workload     |                                                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| workload\_metadata    | {"class": "vision","doc\_url": "``https://github.com/.../README.md``","opset": 7,"type": "body\_analysis","url": "``https://onnxzoo...ferplus.tar.gz``"}   | workload     | source of workload                                                           |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| engine                | tvm / onnx-runtime                                                                                                                                         | compiler     |                                                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| engine\_version       | 0.5 / 996cf30e8d54b4bf58f0c9950475f47bba7e2c7e                                                                                                             | compiler     | include either version or SHA, or both in the format "0.5:996cf3..."         |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| engine\_config        | {“llvm”: “llvm-8”, “nvcc”: 10.1, "accelerator": "MLAS"}                                                                                                    | compiler     | fields are optionally specified                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| compilation\_config   | {"opt\_level": 3, "layer\_schedules": [...]}                                                                                                               | compiler     | optional                                                                     |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| software\_config      | {"os": "ubuntu:18.04","pip": { "docker": "4.1.0", "gitpython": "3.0.4", "numpy": "1.17.4", "onnx": "1.6.0"}}                                               | backend      | env dependency list                                                          |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| hardware\_config      | "mbp13inch2019"                                                                                                                                            | hardware     | descriptor of target hardware environment, optionally json                   |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| runtime\_config       | {“num\_cpu\_threads”: 3, “cudnn”: “cudnn-8”, “cuda\_driver”: “480.10.1”, “os”: linux}                                                                      | hardware     | backend runtime arguments, optionally specified                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| execution\_config     | {“number”: 1, “repeat”: 10, “min\_repeat\_ms”, 0}                                                                                                          | statistics   | workload execution parameters                                                |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| statistics            | {“runtime\_ms\_mean”: 12,“runtime\_ms\_std”: 12}                                                                                                           | statistics   | all metrics, other optional fields ``binary_size``, ``compile_time``, etc.   |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+


Storage format
~~~~~~~~~~~~~~
Currently we're prototyping benchmark data as JSON objects for extensibility and convenience, especially in early versions of the schema. However, as we scale up benchmark aggregation and stabilize parameters, we anticipate switching to a columnar format, such as Arrow or Parquet.

Here is sample data encoded as JSON:

::

   {
      "compilation_config":{
         "relay_opt_lvl":3,
         "tvm_target":"llvm"
      },   
      "engine":"tvm",
      "engine_config":{
         "llvm":"llvm-8"
      },   
      "engine_version":"3486e2c2cf9d0c7c20853f3503528444ce824c1b",
      "hardware_config":"mbp13inch2019",
      "metadata":{
         "docker_tag":"tvmai/ci-gpu:v0.53"
      },   
      "schema_version":0.1,
      "software_config":{
         "os":"ubuntu:18.04",
         "pip":{
            "docker":"4.1.0",
            "gitpython":"3.0.4",
            "numpy":"1.17.4",
            "onnx":"1.6.0"
         }    
      },   
      "statistics":{
         "compile_ms":2848.3831882476807,
         "runtime_ms_mean":237.7542614444445,
         "runtime_ms_std":2.8556173090926698
      },   
      "timestamp":"20191114064203",
      "workload":"vgg19",
      "workload_args":{
         "input_name":"data_0",
         "input_shape":[
         1,   
         3,   
         224, 
         224  
         ]    
      },   
      "workload_metadata":{
         "class":"vision",
         "doc_url":"https://github.com/onnx/models/blob/master/vision/classification/vgg/vgg19/README.md",
         "opset":3,
         "type":"classification",
         "url":"https://s3.amazonaws.com/download.onnx/models/opset_3/vgg19.tar.gz"
      }    
   }
