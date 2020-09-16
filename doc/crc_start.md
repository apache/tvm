# Libcrc API Reference

### CRC start values

| Name | Value (hex) |
| :--- | ---: |
|**`CRC_START_8`**|`00`|
|**`CRC_START_16`**|`0000`|
|**`CRC_START_MODBUS`**|`FFFF`|
|**`CRC_START_XMODEM`**|`0000`|
|**`CRC_START_CCITT_1D0F`**|`1D0F`|
|**`CRC_START_CCITT_FFFF`**|`FFFF`|
|**`CRC_START_KERMIT`**|`0000`|
|**`CRC_START_SICK`**|`0000`|
|**`CRC_START_DNP`**|`0000`|
|**`CRC_START_32`**|`FFFFFFFF`|
|**`CRC_START_64_ECMA`**|`0000000000000000`|
|**`CRC_START_64_WE`**|`FFFFFFFFFFFFFFFF`|

### Description

Before the calculation of the CRC value of a byte blob can start, the value of the CRC must first be initialized. The value to use depends on the CRC algorithm.

### See Also

* [CRC polynomials](crc_poly.md)
