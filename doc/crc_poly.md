# Libcrc API Reference

### CRC polynomial values

| Name | Value (hex) |
| :--- | ---: |
|**`CRC_POLY_16`**|`A001`|
|**`CRC_POLY_32`**|`EDB88320`|
|**`CRC_POLY_64`**|`42F0E1EBA9EA3693`|
|**`CRC_POLY_CCITT`**|`1021`|
|**`CRC_POLY_DNP`**|`A6BC`|
|**`CRC_POLY_KERMIT`**|`8408`|
|**`CRC_POLY_SICK`**|`8005`|

### Description

The mathematical background of CRC is setup with polynomial divisions of a certain order. Each polynomial is represented in the CRC calculation as a bit pattern where each bit defines if a certain order polynomial factor is one or zero. For the algorithms it is enough to know which bit pattern to use. These are defined in the `CRC_POLY_...` constants. For more background you can visit the [CRC calculation page at www.lammertbies.nl](https://www.lammertbies.nl/comm/info/crc-calculation.html).

### See Also

* [CRC start values](crc_start.md)
