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

GETTEXT="_build/locale"
LOCALES="locales"
LANGUAGE=$4

cd $DOCS
echo "$DOCS/$GETTEXT"
if [ ! -d "$DOCS/$GETTEXT" ];
then
    echo "--gettext is not exist, process will automaticly generate, start to execute: make -j$MAXPROCESS gettext."
    make -j$MAXPROCESS gettext
else
    if $1
    then
        echo "--start to generate gettext, execute: make -j20 gettext."
        make -j20 gettext
    else
        echo "--skip generate the gettext."
    fi 
fi

if [ ! -d "$DOCS/$LOCALES" ];
then
    echo "--$DOCS/$LOCALES/ is not exist, process will automaticly generate, start to execute: sphinx-intl update -p $GETTEXT -l $LANGUAGE."
    sphinx-intl update -p $GETTEXT -l $LANGUAGE
else
    if $2
    then
        echo "--start to update *.po, execute: sphinx-intl update -p $GETTEXT -l $LANGUAGE."
        python3 -m sphinx-intl update -p $GETTEXT -l $LANGUAGE
    else
        echo "--skip update the *.po ."
    fi 
fi

if $3
then
    echo "----start to generate html_$LANGUAGE, execute: make -j$MAXPROCESS -e  SPHINXOPTS=\"-D language=\'$LANGUAGE\'\" html  LOCALES=$LOCALES  HTMLFOLD=html_$LANGUAGE"

    # Multi-threading may cause errors
    make -e  SPHINXOPTS="-D language='$LANGUAGE'" html  LOCALES=$LOCALES  HTMLFOLD=html_$LANGUAGE
    # fix github pages theme error
    touch $SCRIPTFOLD/_build/html_$LANGUAGE/.nojekyll

    echo "--finish create the $SCRIPTFOLD/_build/html_$LANGUAGE"
    echo "done."
fi
