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

## Generate multilingual documents

If you have questions or want to get a complete example, we have deployed [a project for Chinese documentation](https://github.com/TVMChinese/tvm). For more information, view the following [repositories](https://github.com/TVMChinese) directly, and this mentioned project is complete and has been deployed on the [website](https://chinese.tvm.wiki/), and our international translation is performed via transifex.com.

0. The following work was done in the following environment:

   * os: https://hub.docker.com/r/tlcpack/ci-gpu
   * tx-version: 0.14.3, py 3.6, x86_64

1. Based on the previous information, prepare the basic environment to ensure that English documents can be generated normally.

2. create gettext fold, where store the `*.pot` and `*.po` files. An automated script has been provided and you can do this easily.

   ```bash
   cd docs/
   ../tests/scripts/task_python_generate_documents.sh true true false <language>

   # for example:
   ../tests/scripts/task_python_generate_documents.sh true true false zh_CN   # for Simplified Chinese, You can find out how to use it at the end of the document.
   ```

   The following types of `language` are supported:

   * bn – Bengali
   * ca – Catalan
   * cs – Czech
   * da – Danish
   * de – German
   * en – English
   * es – Spanish
   * fi – Finnish
   * fr – French
   * hr – Croatian
   * it – Italian
   * lt – Lithuanian
   * nl – Dutch
   * pl – Polish
   * pt_BR – Brazilian Portuguese
   * ru – Russian
   * sl – Slovenian
   * sv – Swedish
   * tr – Turkish
   * uk_UA – Ukrainian
   * zh_CN – Simplified Chinese
   * zh_TW – Traditional Chinese

3. Translate documents directly by translating <*.po>, which is not recommended. We suggest a professional translation platform to implement multi-person collaborative translate-review. If you agree with our advice, you can skip this step. We will use <transifex.com> as an example to tell you how to deploy it.
   
   you can try to translate *.po in `translates/locales/<language>/LC_MESSAGES/`. Notice that `_staging/translates/locales` in fact is a soft-chain to `translates/locales`.

4. Implement cooperative translation of <*.po> with the help of `transifex.com`. An automated script is also provided to help you simplify operations.

   a. config the parameters in `docs/tx_transifex.sh` firstly.

      ```bash
      POTDIR="translates/gettext"                  # the *.pot fold. If you are not sure, default is recommended
      LOCALESDIR="translates/locales"              # the *.po fold, it can be unexist. If you are not sure, default is recommended
      URLNAME=""                                   # you can find it in your project in transifex.com, the same as the share-URL suffix.
      LANGUAGE=""                                  # the target language, "zh_CN" (Simplified Chinese) for example
      ```

   b. init your transifex identity

      ```bash
      cd docs/
      ./tx_transifex.sh init                       # Please feel relaxed to kill the process(Ctrl^C) after input API token and finish verification.
      ```

   c. update your transifex-config, which record the relation between `*.pot` and `*.po`

      ```bash
      cd docs/
      ../tests/scripts/task_python_generate_documents.sh true true false <language>   # update your gettext(*.pot)

      ./tx_transifex.sh update                                                   # update transifex-config
      ```

   d. push your changes to transifex.com

      ```bash
      cd docs/
      ./tx_transifex.sh push
      ```

   e. pull the translated-po files to locale

      ```bash
      cd docs/
      ./tx_transifex.sh pull
      ```

5. generate the html for Chinese.

   We will automatically generate the sitemap(`$DOCS/_build/html_$LANGUAGE/*sitemap.xml`) for you, which is necessary if you expect your site to be detected by search engines like google.  We need you to specify the `ROOTURL` value in `docs/sitemaps/sitemap_generator/config.py`, which represents the url corresponding to the root of the website. If this is not helpful to you, you can skip this step. The default sitemap will still be generated, but it may not be valid.

   ```bash
   cd docs/
   ../tests/scripts/task_python_generate_documents.sh false false true <language> # the _build/html_zh_CN will be created. you can directly deploy to your website.
   ```

6. some tips

   If you are just updating the translation, do the following is enough:

   ```bash
   cd docs/
   ./tx_transifex.sh pull  # if you use <transifex.com> and do translation work remotely, or you can skip this command
   ../tests/scripts/task_python_generate_documents.sh false false true <language>
   ```