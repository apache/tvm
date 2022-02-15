#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# run as: " ./generate_documents.sh [true/false] [true/false] [true/false] [language] "
# the first [true/false]: re-generate gettext or not
# the second [true/false]: update locales file or not
# All above will do their best to satisfy. if not exist, process will generate directly.
# the third [true/false]: generate html or not. The Sitemap will be generate at the same time.
# [language]: the language kind of documents, write as:
# # bn - Bengali
# # ca - Catalan
# # cs - Czech
# # da - Danish
# # de - German
# # en - English
# # es - Spanish
# # fi - Finnish
# # fr - French
# # hr - Croatian
# # it - Italian
# # lt - Lithuanian
# # nl - Dutch
# # pl - Polish
# # pt_BR - Brazilian Portuguese
# # ru - Russian
# # sl - Slovenian
# # sv - Swedish
# # tr - Turkish
# # uk_UA - Ukrainian
# # zh_CN - Simplified Chinese
# # zh_TW - Traditional Chinese

# if you want to add some files to the root of website-fold, you can put then in the docs/html_additions/<language>/

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

MAXPROCESS=4

if [ $# -ne 4 ] 
then
    echo "[Usage]:"
    echo "run as: \" ./generate_documents.sh [true/false] [true/false] [true/false] [language]\""
    echo "for example: \"$0 true true true zh_CN\" can generate documents for Simplified Chinese"
    echo ""
    echo "[Help]:"
    echo "the first [true/false]: re-generate gettext or not"
    echo "the second [true/false]: update locales file or not"
    echo "All above will do their best to satisfy. if not exist, process will generate directly."
    echo "the third [true/false]: generate html or not. The Sitemap will be generate at the same time."
    echo "[language]: the language kind of documents, write as:"
    echo "* bn - Bengali"
    echo "* ca - Catalan"
    echo "* cs - Czech"
    echo "* da - Danish"
    echo "* de - German"
    echo "* en - English"
    echo "* es - Spanish"
    echo "* fi - Finnish"
    echo "* fr - French"
    echo "* hr - Croatian"
    echo "* it - Italian"
    echo "* lt - Lithuanian"
    echo "* nl - Dutch"
    echo "* pl - Polish"
    echo "* pt_BR - Brazilian Portuguese"
    echo "* ru - Russian"
    echo "* sl - Slovenian"
    echo "* sv - Swedish"
    echo "* tr - Turkish"
    echo "* uk_UA - Ukrainian"
    echo "* zh_CN - Simplified Chinese"
    echo "* zh_TW - Traditional Chinese"
    echo "if you want to add some files to the root of website-fold, you can put then in the docs/html_additions/<language>/"
    exit
fi

# calc the docs path
SCRIPTFOLD=$(dirname $0)
DOCS=$(cd "$SCRIPTFOLD/";pwd)

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

    echo "--finish create the $DOCS/_build/html_$LANGUAGE"
    echo "done."

    echo "start to generate sitemap for website. Make sure you have specified the <ROOTURL> in docs/sitemaps/sitemap_generator/config,py, which represents the root path url of the site"
    # old html fold: $DOCS/_build/last_html_$LANGUAGE
    # new html fold: $DOCS/_build/html_$LANGUAGE
    python3 $DOCS/sitemaps/sitemap_generator/generate_sitemap.py --ndir $DOCS/_build/html_$LANGUAGE --odir $DOCS/_build/last_html_$LANGUAGE --ositemap $DOCS/_build/last_html_$LANGUAGE/google-sitemap.xml --sitemap $DOCS/_build/html_$LANGUAGE/google-sitemap.xml

    echo "done"
fi
