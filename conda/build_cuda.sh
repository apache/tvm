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
#/bin/sh
condadir=`dirname $0`
condadir=`readlink -f $condadir`
srcdir=`dirname $condadir`

docker build -t tvm-cuda100-forge $condadir -f $condadir/Dockerfile.cuda100
docker run --rm -v $srcdir:/workspace tvm-cuda100-forge
docker build -t tvm-cuda92-forge $condadir -f $condadir/Dockerfile.cuda92
docker run --rm -v $srcdir:/workspace tvm-cuda92-forge
sudo chown -R `whoami` $condadir/pkg
