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

### With Docker (recommended)

1. Build TVM and the docs inside the [tlcpack/ci-gpu image](https://hub.docker.com/r/tlcpack/ci-gpu) using the [`ci.py`](../tests/scripts/ci.py) script.

   ```bash
   # If this runs into errors, try cleaning your 'build' directory
   python tests/scripts/ci.py docs

   # See other doc building options
   python tests/scripts/ci.py docs --help
   ```

2. Serve the docs and visit http://localhost:8000 in your browser

   ```bash
   # Run an HTTP server you can visit to view the docs in your browser
   python tests/scripts/ci.py serve-docs
   ```

### Native

1. [Build TVM](https://tvm.apache.org/docs/install/from_source.html) first in the repo root folder
2. Install dependencies

   ```bash
   # Pillow on Ubuntu may require libjpeg-dev from apt
   ./docker/bash.sh ci_gpu -c \
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

4. Run an HTTP server and visit http://localhost:8000 in your browser

   ```bash
   cd docs/_build/html && python3 -m http.server
   ```

## Only Execute Specified Tutorials

The document build process will execute all the tutorials in the sphinx gallery.
This will cause failure in some cases when certain machines do not have necessary
environment. You can set `TVM_TUTORIAL_EXEC_PATTERN` to only execute
the path that matches the regular expression pattern.

For example, to only build tutorials under `/vta/tutorials`, run

```bash
python tests/scripts/ci.py docs --tutorials=/vta/tutorials
```

To only build one specific file, do

```bash
# The slash \ is used to get . in regular expression
python tests/scripts/ci.py docs --tutorials=file_name\.py
```

## Helper Scripts

You can run the following script to reproduce the CI sphinx pre-check stage.
This script skips the tutorial executions and is useful to quickly check the content.

```bash
python tests/scripts/ci.py docs --precheck
```

The following script runs the full build which includes tutorial executions.
You will need a GPU CI environment.

```bash
python tests/scripts/ci.py --precheck --full
```

## Define the Order of Tutorials

You can define the order of tutorials with `subsection_order` and
`within_subsection_order` in [`conf.py`](conf.py).
By default, the tutorials within one subsection are sorted by filename.
