# Libcrc API Reference

### `crc_ccitt_1d0f( input_str, num_bytes );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`input_str`**|`const unsigned char *`|The input byte buffer for which the CRC must be calculated|
|**`num_bytes`**|`size_t`|The number of characters in the input buffer|

### Return Value

| Type | Description |
| :--- | :--- |
|`uint16_t`|The resulting CRC value|

### Description

The function `crc_ccitt_1d0f()` calculates a 16 bit CRC value of an input byte buffer based on the CRC calculation algorithm defined by the CCITT with start value `1D0F`.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

Some variations of the CCITT CRC calculations exist with different start values. These are available in the library as separate [`crc_ccitt_ffff()`](crc_ccitt_ffff.md) and [`crc_xmodem()`](crc_xmodem.md).

### See Also

* [`crc_ccitt_ffff();`](crc_ccitt_ffff.md)
* [`crc_xmodem();`](crc_xmodem.md)
* [`update_crc_ccitt();`](update_crc_ccitt.md)
