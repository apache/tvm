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
"""AWS CPU instance target tags."""

from .registry import register_tag


def _register_aws_c5(name, cores, arch):
    register_tag(
        name,
        {
            "kind": "llvm",
            "keys": ["x86", "cpu"],
            "mcpu": arch,
            "num-cores": cores,
        },
    )


_register_aws_c5("aws/cpu/c5.large", 1, "skylake-avx512")
_register_aws_c5("aws/cpu/c5.xlarge", 2, "skylake-avx512")
_register_aws_c5("aws/cpu/c5.2xlarge", 4, "skylake-avx512")
_register_aws_c5("aws/cpu/c5.4xlarge", 8, "skylake-avx512")
_register_aws_c5("aws/cpu/c5.9xlarge", 18, "skylake-avx512")
_register_aws_c5("aws/cpu/c5.12xlarge", 24, "cascadelake")
_register_aws_c5("aws/cpu/c5.18xlarge", 36, "skylake-avx512")
_register_aws_c5("aws/cpu/c5.24xlarge", 48, "cascadelake")
