# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""AWS Trainium target tags."""

from tvm.target import register_tag


def _register_aws_trn1_tag(name, cores):
    register_tag(
        name,
        {
            "kind": "trn",
            "num-cores": cores,
            "partition_size": 128,
            "max_sbuf_size_per_partition": 196608,
            "max_psum_size_per_partition": 16384,
        },
    )


_register_aws_trn1_tag("aws/trn1/trn1.2xlarge", 2)
_register_aws_trn1_tag("aws/trn1/trn1.32xlarge", 32)
