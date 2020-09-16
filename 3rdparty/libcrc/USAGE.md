# Usage of the CRC functions

The source files in the libcrc directory structure contain source code
for functions to calculate several commonly used CRC values: **CRC-8**,
**CRC-16**, **CRC-32**, **CRC-DNP**, **CRC-SICK**, **CRC-Kermit** and **CRC-CCITT**.

The functions can be freely used in open en closed source applications due to the permissive MIT license. The text of
the license can be found in the `LICENSE` file in the root of the
directory tree.

## Calculating a CRC value

A CRC is a single or multi byte value calculated from the contents of
an array of bytes. The following steps must be followed to calculate a CRC:

1. Initialize the CRC value. For CRC-16, CRC-SICK CRC-Kermit and CRC-DNP
the initial value of the CRC is `0x0000`. For CRC-CCITT and CRC-MODBUS,
the value `0xffff` is used. CRC-32 starts with an initial value
of `0xffffffffL`.

2. For each byte of the data starting with the first byte, call the
function `update_crc_8()`, `update_crc_16()`, `update_crc_32()`, `update_crc_dnp()`,
`update_crc_sick()`, `update_crc_kermit()` or `update_crc_ccitt()`
to recalculate the value of the CRC.

3. Only for CRC-32: When all bytes have been processed, take the
one's complement of the obtained CRC value.

4. Only for CRC-DNP: After all input processing, the one's complement
of the CRC is calcluated and the two bytes of the CRC are swapped.

5. Only for CRC-Kermit and CRC-SICK: After all input processing, the
one's complement of the CRC is calcluated and the two bytes of the CRC
are swapped.

## Example program `tstcrc`

An example of this calculation process can be found in the **`tstcrc.c`**
sample source file in the examples subdirectory. This file is automatically compiled to
an executable when the library make process is invoked with **`make`**. In general this example
program and other CRC implementations can be
tested with the test string "**123456789**" without the quotes. The
results should be:

|Type|Result|
| :--- | :--- |
|**CRC16**|`BB3D`|
|**CRC16 Modbus**|`4B37`|
|**CRC16 SICK**|`56A6`|
|**CRC-CCITT**|`31C3` (starting value `0000`)|
|**CRC-CCITT**|`29B1` (starting value `FFFF`)|
|**CRC-CCITT**|`E5CC` (starting value `1D0F`)|
|**CRC-Kermit**|`8921`|
|**CRC-DNP**|`82EA`|
|**CRC32**|`CBF43926`|

The example program **`tstcrc`** can be invoked in three ways:

**`tstcrc -a`**

The program will prompt for an input string. All characters in the
input string are used for the CRC calculation, based on their ASCII
value.

Example input string: **`ABC`**

    CRC16              = 0x4521
    CRC16 (Modbus)     = 0x8550
    CRC16 (Sick)       = 0xC3C1
    CRC-CCITT (0x0000) = 0x3994
    CRC-CCITT (0xffff) = 0xF508
    CRC-CCITT (0x1d0f) = 0x2898
    CRC-CCITT (Kermit) = 0xE359
    CRC-DNP            = 0x5AD3
    CRC32              = 0xA3830348

**`tstcrc -x`**

The program will prompt for an input string. All characters will
be filtered out, except for **0**..**9**, **a**..**f** and **A**..**F**. The remaining characters
will be paired, and every pair of two characters represent the hexadecimal
value to be used for one byte in the CRC calculation. The result if an
odd number of value characters is provided is undefined.

Example input string: **`41 42 43`**

    CRC16              = 0x4521
    CRC16 (Modbus)     = 0x8550
    CRC16 (Sick)       = 0xC3C1
    CRC-CCITT (0x0000) = 0x3994
    CRC-CCITT (0xffff) = 0xF508
    CRC-CCITT (0x1d0f) = 0x2898
    CRC-CCITT (Kermit) = 0xE359
    CRC-DNP            = 0x5AD3
    CRC32              = 0xA3830348

You see, that the result is the same as for the ASCII input "**ABC**". This
is, because **A**, **B** and **C** are represented in ASCII by the hexadecimal
values **41**, **42** and **43**. So it is obvious that the result should be
the same in both cases.

**`tst_crc file1 file2 ...`**

If neither the **`-a`**, nor the **`-x`** parameter is used, the test program
assumes that the parameters are file names. Each file is opened and
the CRC values are calculated.



The newest version of the library source code can be found at Github at
[github.com/lammertb/libcrc/](https://github.com/lammertb/libcrc/)

If you only need the CRC value for a few static inputs, it may be faster
to calculate these CRCs just on-line with a CRC calculater. On-line CRC calculations of CRCs can be performed at
[lammertbies.nl/comm/info/crc-calculation.html](https://www.lammertbies.nl/comm/info/crc-calculation.html)

Support for the CRC routines in this library is provided in the **issues** section on Github at
[github.com/lammertb/libcrc/issues](https://github.com/lammertb/libcrc/issues)

Older support topics can be found at the now read-only Error Detection and Correction forum at
[lammertbies.nl/forum](https://www.lammertbies.nl/forum/viewforum.php?f=11)
