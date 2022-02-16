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

For simplicity, we suggest prioritizing the fields `workload`, `engine`, `hardware` `runtime_ms_mean`, and `runtime_ms_std`. For finer-grained logging, one may additionally propagate the `*_config` fields.

+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| header                | examples                                                                                                                                                                     | category     | notes/justification                                                          |
+=======================+==============================================================================================================================================================================+==============+==============================================================================+
| workload              | resnet-18                                                                                                                                                                    | workload     | name of workload                                                             |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| engine                | "tvm" / "onnxruntime"                                                                                                                                                        | compiler     |                                                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| hardware              | "gcp-c2-standard-16"                                                                                                                                                         | hardware     | descriptor of target hardware environment                                    |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| runtime_ms_mean       | 12.452                                                                                                                                                                       | statistics   |                                                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| runtime_ms_std        | 5.3                                                                                                                                                                          | statistics   |                                                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| timestamp             | 1572282699.6                                                                                                                                                                 | metadata     | indicates when this record is logged                                         |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| schema\_version       | "0.1"                                                                                                                                                                        | metadata     | ensure reproducibility as we iterate on this schema                          |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| metadata              | { "docker\_tag":"gcr.io/.../0a680", ... }                                                                                                                                    | metadata     | ``docker_tag`` is optional                                                   |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| workload\_args        | {“input\_name”: "Input3", “input\_shape”: [list\_of\_shape], “data\_layout”: NHCW}                                                                                           | workload     |                                                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| workload\_metadata    | {"class": "vision","doc\_url": "``https://github.com/.../README.md``", "opset": 7,"type": "body\_analysis","url": "``https://onnxzoo...ferplus.tar.gz``", "md5": "07fc7..."} | workload     | source of workload                                                           |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| engine\_version       | "1.0.5"                                                                                                                                                                      | compiler     | use semvar format                                                            |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| engine\_config        | {“llvm”: “llvm-8”, “nvcc”: 10.1, "accelerator": "MLAS", "relay_opt_level": 3, "tvm_target":"llvm -mcpu=cascadelake"}                                                         | compiler     | fields are optionally specified                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| compilation\_config   | {"opt_level": 3, "layer_schedules":[]/ <SHA_to_schedules>}                                                                                                                   | compiler     | fields are optionally specified                                              |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| software\_config      | {"os": "ubuntu:18.04","pip": { "docker": "4.1.0", "gitpython": "3.0.4", "numpy": "1.17.4", "onnx": "1.6.0"}, “cudnn”: “cudnn-8”, "cuda_driver”: “480.10.1”}                  | backend      | env dependency list                                                          |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| runtime\_config       | {"num_cpu_threads": 3}                                                                                                                                                       | backend      | info on non-hardware, non-software metadata                                  |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| hardware\_config      | {"cpu_count": 16, "cloud_machine_type":"c2-standard-16", "memory_GB":64}                                                                                                     | hardware     | json descriptor of target hardware environment                               |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| execution\_config     | {“number”: 1, “repeat”: 10, “min\_repeat\_ms”, 0}                                                                                                                            | statistics   | workload execution parameters                                                |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| metrics               | {“accuracy”: 48.5,“compilation_ms_mean”: 12}                                                                                                                                 | statistics   | other metrics                                                                |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+
| runtime_raw           | [{"runtime_ms": 12, ...}, {"runtime_ms":13,...},...]                                                                                                                         | statistics   | optional raw metrics array                                                   |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------+



Storage format
~~~~~~~~~~~~~~
Currently we're prototyping benchmark data as JSON objects for extensibility and convenience, especially in early versions of the schema. However, as we scale up benchmark aggregation and stabilize parameters, we anticipate switching to a columnar format, such as Arrow or Parquet.

Here is sample data encoded as JSON:

::

  {
    "workload":"arcface_resnet100",
    "engine":"tvm",
    "hardware":"gcp-c2-standard-16",
    "runtime_ms_mean":109.43004820081924,
    "runtime_ms_std":0.09078385126800587,
    "timestamp":"20191123003411",
    "schema_version":"0.1",
    "metadata":{
      "docker_tag":"tlcpack/ci-gpu:v0.53"
    },
    "workload_args":{
      "input_shape_dict":{
        "data":[
          1,
          3,
          112,
          112
        ]
      },
      "input_type_dict":{
        "data":"float32"
      },
      "input_value_dict":{}
    },
    "workload_metadata":{
      "class":"vision",
      "doc_url":"https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/README.md",
      "md5":"66074b860f905295aab5a842be57f37d",
      "opset":8,
      "type":"body_analysis",
      "url":"https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz"
    },
    "engine_version":"1.0.0",
    "engine_config":{},
    "compilation_config":{
      "relay_opt_level": 3
    },
    "software_config":{
      "os":"ubuntu:18.04",
      "pip":{
        "docker":"4.1.0",
        "gitpython":"3.0.4",
        "numpy":"1.17.4",
        "onnx":"1.6.0"
      }
    },
    "runtime_config":{},
    "hardware_config":{
      "cloud_machine_type":"c2-standard-16",
      "cloud_provider":"GCP",
      "cpu_count":16,
      "cpu_platform":"Intel Cascade Lake",
      "memory_GB":64
    },
    "execution_config":{},
    "metrics":{}
  }
