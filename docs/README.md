<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM Documentation
This folder contains the source of TVM's documentation, hosted at https://tvm.apache.org/docs

## Build Locally

See also the instructions below to run a specific tutorial. Note that some of the tutorials need GPU support. Once build, either of these can be served using Python's build in HTTP server:

```bash
# Run this and then visit http://localhost:8000 in your browser
cd docs/_build/html && python3 -m http.server
```

### With Docker (recommended)

1. Build TVM and the docs inside the [tlcpack/ci-gpu image](https://hub.docker.com/r/tlcpack/ci-gpu)

    ```bash
    # If this runs into errors, try cleaning your 'build' directory
    make docs
    ```


### Native

1. [Build TVM](https://tvm.apache.org/docs/install/from_source.html) first in the repo root folder
2. Install dependencies

    ```bash
    # Pillow on Ubuntu may require libjpeg-dev from apt
    docker run tlcpack/ci-gpu:v0.78 bash -c \
        'python3 -m pip install --quiet tlcpack-sphinx-addon==0.2.1 synr==0.5.0 && python3 -m pip freeze' > frozen-requirements.txt

    pip install -r frozen-requirements.txt
    ```

3. Generate the docs

    ```bash
    # TVM_TUTORIAL_EXEC_PATTERN=none skips the tutorial execution to the build
    # work on most environments (e.g. MacOS).
    export TVM_TUTORIAL_EXEC_PATTERN=none

    make html
    ```


## Only Execute Specified Tutorials
The document build process will execute all the tutorials in the sphinx gallery.
This will cause failure in some cases when certain machines do not have necessary
environment. You can set `TVM_TUTORIAL_EXEC_PATTERN` to only execute
the path that matches the regular expression pattern.

For example, to only build tutorials under `/vta/tutorials`, run

```bash
TVM_TUTORIAL_EXEC_PATTERN=/vta/tutorials make html
```

To only build one specific file, do

```bash
# The slash \ is used to get . in regular expression
TVM_TUTORIAL_EXEC_PATTERN=file_name\.py make html
```

## Helper Scripts

You can run the following script to reproduce the CI sphinx pre-check stage.
This script skips the tutorial executions and is useful to quickly check the content.

```bash
./tests/scripts/task_sphinx_precheck.sh
```

The following script runs the full build which includes tutorial executions.
You will need a GPU CI environment.

```bash
python tests/scripts/ci.py --precheck --full
```

## Define the Order of Tutorials
You can define the order of tutorials with `conf.py::subsection_order` and `conf.py::within_subsection_order`.
By default, the tutorials within one subsection is sorted by filename.
