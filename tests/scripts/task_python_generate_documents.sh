#!/bin/bash

# run as: " ./task_python_generate_documents.sh [0/1] [0/1] [language] "
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

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

MAXPROCESS=20

if [ $# -ne 4 ] 
then
    echo "error bash command, please run as: \" ./task_python_generate_documents.sh [true/false] [true/false] [true/false] [language]\""
    echo "for example: \"$0 true true true zh_CN\" can generate documents for Chinese"
    exit
fi

# calc the docs path
SCRIPTFOLD=$(dirname $0)
DOCS=$(cd "$SCRIPTFOLD/../../docs/";pwd)

BUILDDIR="_build"
TXDIR=".tx"
GETTEXT="translates/gettext"
LOCALE="translates/locale"
LOCALES="translates/locales"
LANGUAGE=$4

cd $DOCS
if [ ! -d "$DOCS/$GETTEXT" ];
then
    echo "--gettext is not exist, process will automaticly generate, start to execute: make -j$MAXPROCESS gettext."
    make -j$MAXPROCESS gettext
else
    if $1
    then
        echo "--start to generate gettext, execute: make -j20 gettext."
        make -j$MAXPROCESS gettext
    else
        echo "--skip generate the gettext."
    fi 
fi

if [ ! -d "$DOCS/$LOCALES/$LANGUAGE" ];
then
    echo "--$DOCS/$LOCALES/$LANGUAGE is not exist, process will automaticly generate, start to execute: sphinx-intl update -p $GETTEXT -l $LANGUAGE."
    pip3 install sphinx-intl
    sphinx-intl update -p $GETTEXT -l $LANGUAGE

    mkdir -p $DOCS/$BUILDDIR
    ln -s $DOCS/$GETTEXT $DOCS/$BUILDDIR/locale
else
    if $2
    then
        echo "--start to update *.pot, execute: sphinx-intl update -p $GETTEXT -l $LANGUAGE."
        pip3 install sphinx-intl
        sphinx-intl update -p $GETTEXT -l $LANGUAGE
        
        mkdir -p $DOCS/$BUILDDIR
        ln -s $DOCS/$GETTEXT $DOCS/$BUILDDIR/locale
    else
        echo "--skip update the *.pot ."
    fi 
fi

if $3
then
    echo "----start to generate html_$LANGUAGE, execute: make -j$MAXPROCESS -e  SPHINXOPTS=\"-D language=\'$LANGUAGE\'\" html  LOCALES=$LOCALES  HTMLFOLD=html_$LANGUAGE"
    mkdir -p $DOCS/html_additions/$LANGUAGE/

    # backup http website
    rm -r $DOCS/_build/last_html_$LANGUAGE
    mv $DOCS/_build/html_$LANGUAGE $DOCS/_build/last_html_$LANGUAGE

    # Multi-threading may cause errors
    make -e  SPHINXOPTS="-D language='$LANGUAGE'" html  LOCALES=$LOCALES  HTMLFOLD=html_$LANGUAGE
    # fix github pages theme error
    touch $DOCS/_build/html_$LANGUAGE/.nojekyll
    cp -r $DOCS/html_additions/$LANGUAGE/* $DOCS/_build/html_$LANGUAGE/

    echo "if you want to flush the contribute pages, please download and replace docs/contribute_translate/translator_data.csv"
    python3 $DOCS/contribute_translate/build_html.py
    cp $DOCS/contribute_translate/declaration_zh_CN.html $DOCS/_build/html_$LANGUAGE/

    echo "--finish create the $DOCS/_build/html_$LANGUAGE"
    echo "done."

    echo "start to generate sitemap for website."
    # old html fold: $DOCS/_build/last_html_$LANGUAGE
    # new html fold: $DOCS/_build/html_$LANGUAGE
    python3 $DOCS/sitemaps/sitemap_generator/generate_sitemap.py --ndir $DOCS/_build/html_$LANGUAGE --odir $DOCS/_build/last_html_$LANGUAGE --ositemap $DOCS/_build/last_html_$LANGUAGE/google-sitemap.xml --sitemap $DOCS/_build/html_$LANGUAGE/google-sitemap.xml

    echo "done"
fi
