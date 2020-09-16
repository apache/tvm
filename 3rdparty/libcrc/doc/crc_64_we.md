# Libcrc API Reference

### `crc_64_we( input_str, num_bytes );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`input_str`**|`const unsigned char *`|The input byte buffer for which the CRC must be calculated|
|**`num_bytes`**|`size_t`|The number of characters in the input buffer|

### Return Value

| Type | Description |
| :--- | :--- |
|`uint64_t`|The resulting CRC value|

### Description

The function `crc_64_we()` calculates a 64 bit CRC value of an input byte buffer based on the 64 bit CRC calculation algorithm used by Wolfgang Ehrhardt in his FAR Manager plugin. The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

### See Also

* [`crc_64_ecma();`](crc_64_ecma.md)
* [`update_crc_64();`](update_crc_64.md)
