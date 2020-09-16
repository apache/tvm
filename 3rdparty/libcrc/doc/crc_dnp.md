# Libcrc API Reference

### `crc_dnp( input_str, num_bytes );`

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

The function `crc_dnp()` calculates a 16 bit CRC value of an input byte buffer based on the CRC calculation algorithm used in the DNP protocol.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

### See Also

* [`update_crc_dnp();`](update_crc_dnp.md)
