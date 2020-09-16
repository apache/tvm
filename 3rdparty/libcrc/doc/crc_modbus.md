# Libcrc API Reference

### `crc_modbus( input_str, num_bytes );`

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

The function `crc_modbus()` calculates a 16 bit CRC value of an input byte buffer based on the CRC calculation algorithm used by the Modbus protocol.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

The Modbus CRC calculation uses the same polynomial as the standard CRC-16, but with a different initialization value.

### See Also

* [`crc_16();`](crc_16.md)
* [`update_crc_16();`](update_crc_16.md)
