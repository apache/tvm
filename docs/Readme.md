#### How to generate Chinese-Docs
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
