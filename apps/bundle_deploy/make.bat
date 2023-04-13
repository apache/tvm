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

@echo off

set WORK_DIR=%~dp0
set BUILD_DIR=%WORK_DIR%\build
set TVM_NUM_THREADS=1

where /Q cmake
if ERRORLEVEL 1 (
    echo [^!] Couldn't find cmake.exe in system, did you install cmake?
    exit /b 1
)

where /Q python
if ERRORLEVEL 1 (
    echo [^!] Couldn't find python.exe in system, did you install python?
    exit /b 1
)

python -c "import tvm"
if ERRORLEVEL 1 (
    echo [^!] Couldn't import tvm in python, did you install tvm?
    exit /b 1
)

:: set the python executable of current environment, or cmake may find another
:: interpreter in the system (e.g. another Conda env)
for /F %%i in ('python -c "import sys;print(sys.executable)"') do (
    set PYTHON=%%i
)

if not exist "%BUILD_DIR%" (
    mkdir %BUILD_DIR%
)

:argparse
if not "%1" == "" (
    set CONSUMED=0
    if /I "%1" == "clean"        ( call :clean && exit /b %ERRORLEVEL% )
    if /I "%1" == "cleanall"     ( call :cleanall && exit /b %ERRORLEVEL% )
    if /I "%1" == "demo_dynamic" ( call :demo_dynamic && exit /b %ERRORLEVEL% )
    if /I "%1" == "test_dynamic" ( call :test_dynamic && exit /b %ERRORLEVEL% )
    if /I "%1" == "demo_static"  ( call :demo_static && exit /b %ERRORLEVEL% )
    if /I "%1" == "test_static"  ( call :test_static && exit /b %ERRORLEVEL% )
    if "!CONSUMED!" == "0"       ( echo [^!] Unknown parameter: "%1" )
    shift
    goto :argparse
)
exit /b 0

:demo_dynamic
cmake -DPython_EXECUTABLE="%PYTHON%" -B"%BUILD_DIR%" -S"%WORK_DIR%"
cmake --build "%BUILD_DIR%" --config Release --target bundle demo_dynamic -- /m

if not ERRORLEVEL 1 (
    "%BUILD_DIR%\Release\demo_dynamic.exe" ^
    "%BUILD_DIR%\Release\bundle.dll" ^
    "%BUILD_DIR%\graph_cpp.json" ^
    "%BUILD_DIR%\params_cpp.bin" ^
    "%BUILD_DIR%\cat.bin"
)
exit /b %ERRORLEVEL%

:test_dynamic
cmake -DPython_EXECUTABLE="%PYTHON%" -B"%BUILD_DIR%" -S"%WORK_DIR%"
cmake --build "%BUILD_DIR%" --config Release --target test_bundle test_dynamic -- /m

if not ERRORLEVEL 1 (
    "%BUILD_DIR%\Release\test_dynamic.exe" ^
    "%BUILD_DIR%\Release\test_bundle.dll" ^
    "%BUILD_DIR%\test_data_cpp.bin" ^
    "%BUILD_DIR%\test_output_cpp.bin" ^
    "%BUILD_DIR%\test_graph_cpp.json" ^
    "%BUILD_DIR%\test_params_cpp.bin"
)
exit /b %ERRORLEVEL%

:demo_static
echo [^!] Windows doesn't support static bundle deploy for now.
exit /b 1

:test_static
echo [^!] Windows doesn't support static bundle deploy for now.
exit /b 1

:clean
del /Q /F "%BUILD_DIR%\Release\bundle.dll" "%BUILD_DIR%\Release\test_bundle.dll"
exit /b %ERRORLEVEL%

:cleanall
rmdir /Q /S "%BUILD_DIR%"
exit /b %ERRORLEVEL%
