# Libcrc API Reference

### `crc_16( input_str, num_bytes );`

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

The function `crc_16()` calculates a 16 bit CRC value of an input byte buffer based on the common 16 bit CRC calculation algorithm with start value `0000`.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

Note that the Modbus CRC calculation available in the function [`crc_modbus()`](crc_modbus.md) uses the same algorithm but with a different initialization value for the CRC.

### See Also

* [`crc_modbus();`](crc_modbus.md)
* [`update_crc_16();`](update_crc_16.md)
