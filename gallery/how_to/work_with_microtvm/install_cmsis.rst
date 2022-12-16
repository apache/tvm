Install CMSIS-NN
----------------------------

    .. code-block:: bash

        %%shell
        CMSIS_SHA="51263182d16c92649a48144ba56c0945f9fce60e"
        CMSIS_SHASUM="d02573e5a8908c741d8558f01be2939aae6e940933ccb58123fa972864947759eefe5d554688db3910c8ed665a248b477b5e4458e12773385c67f8a2136b3b34"
        CMSIS_URL="http://github.com/ARM-software/CMSIS_5/archive/${CMSIS_SHA}.tar.gz"
        export CMSIS_PATH=/root/cmsis
        DOWNLOAD_PATH="/root/${CMSIS_SHA}.tar.gz"
        mkdir ${CMSIS_PATH}
        wget ${CMSIS_URL} -O "${DOWNLOAD_PATH}"
        tar -xf "${DOWNLOAD_PATH}" -C ${CMSIS_PATH} --strip-components=1
        rm ${DOWNLOAD_PATH}
