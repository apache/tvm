# Libcrc API Reference

### `crc_xmodem( input_str, num_bytes );`

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

The function `crc_xmodem()` calculates a 16 bit CRC value of an input byte buffer based on the CRC calculation algorithm used in the Xmodem protocol.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

The Xmodem CRC calculation is a variant of the CCITT CRC calculation algorithm but with a different initialization parameter.

### See Also

* [`crc_ccitt_1d0f();`](crc_ccitt_1d0f.md)
* [`crc_ccitt_ffff();`](crc_ccitt_ffff.md)
* [`update_crc_ccitt();`](update_crc_ccitt.md)
