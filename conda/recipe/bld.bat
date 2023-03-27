:: Licensed to the Apache Software Foundation (ASF) under one
:: or more contributor license agreements.  See the NOTICE file
:: distributed with this work for additional information
:: regarding copyright ownership.  The ASF licenses this file
:: to you under the Apache License, Version 2.0 (the
:: "License"); you may not use this file except in compliance
:: with the License.  You may obtain a copy of the License at
::
::   http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing,
:: software distributed under the License is distributed on an
:: "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
:: KIND, either express or implied.  See the License for the
:: specific language governing permissions and limitations
:: under the License.
echo on

rd /s /q build
mkdir build
cd build

cmake ^
      -G "Visual Studio 16 2019" ^
      -DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
      -DCMAKE_INSTALL_PREFIX:PATH=%LIBRARY_PREFIX% ^
      -DUSE_LLVM=ON ^
      -DUSE_RPC=ON ^
      -DUSE_CPP_RPC=ON ^
      -DUSE_MICRO=ON ^
      -DUSE_SORT=ON ^
      -DUSE_RANDOM=ON ^
      -DUSE_PROFILER=ON ^
      -DINSTALL_DEV=ON ^
      %SRC_DIR%

cd ..
:: defer build to install stage to avoid rebuild.
:: sometimes windows msbuild is not very good at file
:: caching and install will results in a rebuild
