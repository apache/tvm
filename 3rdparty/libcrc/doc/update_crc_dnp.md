# Libcrc API Reference

### `update_crc_dnp( crc, c );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`crc`**|`uint16_t`|The CRC value calculated from the byte stream upto but not including the current byte|
|**`c`**|`unsigned char`|The next byte from the byte stream to be used in the CRC calculation|

### Return Value

| Type | Description |
| :--- | :--- |
|**`uint16_t`**|The new CRC value of the byte stream including the current byte|

### Description

The function `update_crc_dnp()` can be used to calculate the CRC value in a stream of bytes where it is not possible to first buffer the stream completely to calculate the CRC when all data is received. The parameters are the previous CRC value and the current byte which must be used to calculate the new CRC value.

In order for this function to work properly, the CRC value must be initialized before the first call to `update_crc_dnp()`. The DNP protocol needs the initialization value `CRC_START_DNP` to do the proper CRC calculation.

Please note that when all bytes have been processed that all bits of the resulting CRC must be inverted and that the high and low order byte must be swapped in order to get the final result. An example of this can be found in the function `crc_dnp()` in the source file [`crcdnp.c`](../src/crcdnp.c).

### See Also

* [`crc_dnp();`](crc_dnp.md)
* [CRC start values](crc_start.md)
