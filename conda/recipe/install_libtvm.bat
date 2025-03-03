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

cmake --build build --config Release --target install || exit /b

:: Copy files into library bin so that they can be found
cp %LIBRARY_LIB%\tvm.dll %LIBRARY_BIN%\tvm.dll || exit /b
cp %LIBRARY_LIB%\tvm_runtime.dll %LIBRARY_BIN%\tvm_runtime.dll || exit /b
