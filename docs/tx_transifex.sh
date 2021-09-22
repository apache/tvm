# tx_transifex.sh [init]/[update]/[push]/[pull]

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPTFOLD="$( cd "$( dirname "$0"  )" && pwd )"

POTDIR="translates/gettext"                 # the *.pot fold
LOCALESDIR="translates/locales"             # the *.po fold, it can be unexist.
URLNAME="tvmchinese_0_8_dev"                # depend on transifex.com 
LANGUAGE="zh_CN"

cd $SCRIPTFOLD
if [ $# -ne 1 ] 
then
    echo "error bash command, please run as: \" ./tx_transifex.sh [init]/[update]/[push]/[pull]\""
    exit
fi

if [ "$1" == "--help" ];
then
    echo "run as: \" ./tx_transifex.sh [init]/[update]/[push]/[pull]\""
    exit
fi

if [ "$1" == "init" ];
then
    pip3 install transifex-client
    echo "\"tx init\" will be executed. After input and finish verifying the API Token, you can interrupt the program directly by Ctrl^C!"
    tx init
    echo "done."
    exit
fi

if [ "$1" == "update" ];
then
    echo "start to generate/update the config. make sure you have config this script's \$POTDIR and \$LOCALESDIR"
    mkdir -p $SCRIPTFOLD/.tx/translates

    ln -s $SCRIPTFOLD/$POTDIR $SCRIPTFOLD/.tx/$POTDIR
    sphinx-intl update-txconfig-resources --pot-dir $POTDIR --transifex-project-name $URLNAME -d $LOCALESDIR

    echo "you can ignore the \"tx set: error: the following arguments are required: -r/--resource, filename\", this script will fix it automaticly."

    sed -i '/type.*PO.*$/d' $SCRIPTFOLD/.tx/config && sed -i 's/source_lang.*$/source_lang = en\ntype = PO/' $SCRIPTFOLD/.tx/config     # fix the no-type bug of sphinx-intl update-txconfig-resources
    
    echo "done."
    exit
fi

if [ "$1" == "push" ];
then
    echo "push to transifex.com, according to .tx/config"
    cd $SCRIPTFOLD/.tx/ && tx push -s
    echo "done."
    exit
fi

if [ "$1" == "pull" ];
then
    echo "start to pull <$LANGUAGE>*.po from transifex.com..."

    rm -r $SCRIPTFOLD/$LOCALESDIR/old_$LANGUAGE
    mv $SCRIPTFOLD/$LOCALESDIR/$LANGUAGE $SCRIPTFOLD/$LOCALESDIR/old_$LANGUAGE

    cd $SCRIPTFOLD/.tx/ && tx pull -l $LANGUAGE
    
    echo "done."
    exit
fi