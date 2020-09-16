# Libcrc API Reference

### `crc_ccitt_ffff( input_str, num_bytes );`

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

The function `crc_ccitt_ffff()` calculates a 16 bit CRC value of an input byte buffer based on the CRC calculation algorithm defined by the CCITT with start value `FFFF`.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

Some variations of the CCITT CRC calculation exist where different start values are used. These are available in the library as [`crc_ccitt_1d0f()`](crc_ccitt_1d0f.md) and [`crc_xmodem()`](crc_xmodem.md).

### See Also

* [`crc_ccitt_1d0f();`](crc_ccitt_1d0f.md)
* [`crc_xmodem();`](crc_xmodem.md)
* [`update_crc_ccitt();`](update_crc_ccitt.md)
