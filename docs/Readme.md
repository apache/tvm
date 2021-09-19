### How to generate Chinese-Docs
0. prepare the env, only give a kind of way.

* prepare the container

```bash
# install docker first.
docker pull tlcpack/ci-gpu:v0.77

# then configure LLVM, CUDA and TVM environments as usual, use /home/syfeng/tvm-upstream/tests/scripts/task_config_build_gpu.sh to create config.cmake
```

* in a new tvm fold to compile the tvm again, use `/home/syfeng/tvm-upstream/tests/scripts/task_config_build_gpu.sh` to `create config.cmake` and start your work

* Consider using proxy to enhance your network environment, which may help a lot.

1. create gettext fold, where store the `*.pot` and `*.po` files.

```bash
cd docs/
../tests/scripts/task_python_generate_documents.sh true true false zh_CN 

# you can also make docs for other kinds of language. for example:
../tests/scripts/task_python_generate_documents.sh true true false fr   # for French, You can find out how to use it at the end of the document.
```

2. if you want to translate your document without the help of transifex.com, you can try to translate *.po in `translates/locales/<language>/LC_MESSAGES/`. Notice that `_staging/translates/locales` in fact is a soft-chain to `translates/locales`. if you plan to take the advantage of transifex.com, just skip this step.

3. this step will connect your project to transifex.com, it's optional.

* config the parameters in `docs/tx_transifex.sh` firstly.

```bash
POTDIR="translates/gettext"                 # the *.pot fold. If you are not sure, default is recommended
LOCALESDIR="translates/locales"             # the *.po fold, it can be unexist. If you are not sure, default is recommended
URLNAME="URL_suffix"                        # you can find it in your project in transifex.com, the same as the share-URL suffix.
LANGUAGE="zh_CN"                            # the aim-language
```

* init your transifex identity

```bash
cd docs/
./tx_transifex.sh init                      # Please feel free to kill the process(Ctrl^C) after input API token and finish verification.
```

* update your transifex-config, which record the relation between `*.pot` and `*.po`

```bash
cd docs/
../tests/scripts/task_python_generate_documents.sh true false false zh_CN   # update your gettext(*.pot)

./tx_transifex.sh update                                                    # update transifex-config
```

* push your changes to transifex.com

```bash
cd docs/
./tx_transifex.sh push
```

* pull the translated-po files to locale

```bash
cd docs/
./tx_transifex.sh pull
```

4. generate the html for Chinese.

```bash
cd docs/
../tests/scripts/task_python_generate_documents.sh false false true zh_CN # the _build/html_zh_CN will be created. you can directly deploy to your website.
```

Introduction about usage of `tests/scripts/task_python_generate_documents.sh`

```bash
# run tests/scripts/task_python_generate_documents.sh can automaticly finish this work

# run as: " ./task_python_generate_documents.sh [true/false] [true/false] [true/false] [language] "
# the first [true/false]: re-generate gettext or not
# the second [true/false]: update locales file or not
# the third [true/false]: generate html or not
# The two above will do their best to satisfy. if not exist, process will generate directly.
# [language]: the language kind of documents, write as:
# # bn – Bengali
# # ca – Catalan
# # cs – Czech
# # da – Danish
# # de – German
# # en – English
# # es – Spanish
# # fi – Finnish
# # fr – French
# # hr – Croatian
# # it – Italian
# # lt – Lithuanian
# # nl – Dutch
# # pl – Polish
# # pt_BR – Brazilian Portuguese
# # ru – Russian
# # sl – Slovenian
# # sv – Swedish
# # tr – Turkish
# # uk_UA – Ukrainian
# # zh_CN – Simplified Chinese
# # zh_TW – Traditional Chinese
```


<!-- 
#### The following document show the original method, it has been abondoned
1. generate `gettext` fold

    ```bash
    cd docs/
    make -j20 gettext

    # you will find a fold named "locale" generated in docs/_build/ 
    ```

2. create `*.po` files

    ```bash
    cd docs/_staging/
    sphinx-intl update -p _build/locale/ -l zh_CN -l en   # It will set up internationalization support for English and Chinese

    # then docs/_staging/locale will be generated. For the Chinese manual, we only need to translate docs/_staging/locale/zh_CN/LC_MESSAGES/*.po
    # Of course, when run "make clean", the locale should not be delete. 
    ```

3. generate static-website-code for multi-language (Chinese for example):

    ```bash
    make -e  SPHINXOPTS="-D language='zh_CN'" html ./source build/html-zh

    # the fold "html", written in Chinese, will be generated in docs/_build/. We can deploy it normally
    ``` 
-->
