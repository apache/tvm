#### How to generate Chinese-Docs

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

./task_python_generate_documents.sh false true true zh_CN      # may help you generate Chinese manual     
```

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
